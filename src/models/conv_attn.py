# -*- coding: utf-8 -*-
from typing import Optional, Literal, Tuple, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Hugging Face Phi-3 ----
from transformers.models.phi3.configuration_phi3 import Phi3Config
from transformers.models.phi3.modeling_phi3 import (
    Phi3PreTrainedModel,
    Phi3Attention,
    Phi3MLP,
    Phi3RMSNorm,
)
from transformers.modeling_outputs import BaseModelOutputWithPast


# =========================
# Config
# =========================
class PyramidPhi3Config(Phi3Config):
    """
    偶数層: 前半が DownScale、後半が UpScale
    - window_sizes / strides: 各 Down/Up レベルごと（長さ = num_hidden_layers//2）
      省略時は全レベル同一
    - summary_type: 'attn' or 'mean'（窓→ブロック要約）
    - up_mode: 'overlap_add' or 'cross_attn'（UpScale 復元方法）
    - learnable_kernel: overlap_add の窓カーネルを学習するか
    - strict_causal: UpScale 復元時に「ブロック終端 ≤ t」制約を課す
    - post_attention_dense_repeats: 各段の MLP 反復回数
    """
    model_type = "pyramid_phi3"

    def __init__(
        self,
        window_sizes: Optional[List[int]] = None,
        window_strides: Optional[List[int]] = None,
        summary_type: str = "attn",
        up_mode: str = "overlap_add", ## or "cross_attn"
        learnable_kernel: bool = False,
        strict_causal: bool = True,
        post_attention_dense_repeats: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.summary_type = summary_type
        self.up_mode = up_mode
        self.learnable_kernel = learnable_kernel
        self.strict_causal = strict_causal
        self.post_attention_dense_repeats = post_attention_dense_repeats

        # Down/Up の段数
        n_half = max(1, kwargs.get("num_hidden_layers", 2) // 2)
        if window_sizes is None:
            window_sizes = [100] * n_half
        if window_strides is None:
            window_strides = [5] * n_half
        assert len(window_sizes) == n_half and len(window_strides) == n_half, \
            "window_sizes/strides must have length = num_hidden_layers//2"
        self.window_sizes = window_sizes
        self.window_strides = window_strides


# =========================
# Utils
# =========================
def _triangular_kernel(W: int, device, dtype):
    up = torch.linspace(0, 1, steps=(W // 2) + 1, device=device, dtype=dtype)
    left = up[:-1]
    right = torch.flip(up, dims=[0])
    w = torch.cat([left, right], dim=0)
    if w.numel() < W:
        w = F.pad(w, (0, W - w.numel()), value=w[-1])
    return w  # [W]

def _lower_tri_mask(L: int, device, dtype):
    m = torch.full((L, L), torch.finfo(dtype).min, device=device, dtype=dtype)
    m = torch.triu(m, diagonal=1)
    return m.view(1, 1, L, L)

def _build_overlap_matrix(S: int, N: int, W: int, R: int, device, dtype, kernel: torch.Tensor, strict_causal: bool):
    A = torch.zeros(S, N, device=device, dtype=dtype)
    starts = torch.arange(N, device=device) * R
    ends = starts + (W - 1)
    for i in range(N):
        s_i, e_i = int(starts[i].item()), int(ends[i].item())
        t0, t1 = max(0, s_i), min(S - 1, e_i)
        if t0 > t1:
            continue
        if strict_causal:
            t0_eff = max(t0, e_i)
            if t0_eff <= t1:
                idx = torch.arange(t0_eff, t1 + 1, device=device)
                A[idx, i] = kernel[idx - s_i]
        else:
            idx = torch.arange(t0, t1 + 1, device=device)
            A[idx, i] = kernel[idx - s_i]
    denom = A.sum(dim=1, keepdim=True)
    A = A / (denom + 1e-8)
    return A  # [S,N]

def _block_ends(N: int, W: int, R: int, device):
    return (torch.arange(N, device=device) * R) + (W - 1)  # [N]


# =========================
# ③ Token←Block Cross-Attn
# =========================
class TokenBlockCrossAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, bias: bool = False):
        super().__init__()
        self.H = hidden_size
        self.h = num_heads
        self.d = hidden_size // num_heads
        assert hidden_size % num_heads == 0
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

    def forward(self, token_q: torch.Tensor, block_kv: torch.Tensor, block_ends: torch.Tensor, strict_causal: bool = True):
        B, S, H = token_q.shape
        B2, N, H2 = block_kv.shape
        assert B == B2 and H == H2
        q = self.q_proj(token_q).view(B, S, self.h, self.d).transpose(1, 2)  # [B,h,S,d]
        k = self.k_proj(block_kv).view(B, N, self.h, self.d).transpose(1, 2) # [B,h,N,d]
        v = self.v_proj(block_kv).view(B, N, self.h, self.d).transpose(1, 2) # [B,h,N,d]
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d)       # [B,h,S,N]
        if strict_causal:
            device, dtype = token_q.device, token_q.dtype
            t_idx = torch.arange(S, device=device).view(1,1,S,1)
            ends = block_ends.view(1,1,1,N)
            mask = (ends <= t_idx)
            attn = attn.masked_fill(~mask, torch.finfo(dtype).min)
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S, H)
        return self.o_proj(out)


# =========================
# Up-Scaler（①/③ 切替）
# =========================
class Upscaler(nn.Module):
    def __init__(self, hidden_size: int, W: int, R: int,
                 mode: Literal["overlap_add", "cross_attn"] = "overlap_add",
                 learnable_kernel: bool = False, strict_causal: bool = True,
                 num_heads: int = 8):
        super().__init__()
        self.H, self.W, self.R = hidden_size, W, R
        self.mode, self.strict_causal = mode, strict_causal
        if learnable_kernel:
            self.kernel = nn.Parameter(torch.randn(W))
        else:
            self.register_buffer("kernel", None, persistent=False)
        self.learnable_kernel = learnable_kernel
        self.cross = TokenBlockCrossAttention(hidden_size, num_heads=num_heads)

    def _kernel(self, device, dtype):
        if self.learnable_kernel:
            w = F.relu(self.kernel.to(device=device, dtype=dtype))
            if w.sum() > 0:
                w = w / (w.sum() + 1e-8) * self.W
            return w
        return _triangular_kernel(self.W, device, dtype)

    def forward(self, Z: torch.Tensor, S_out: int, token_q: Optional[torch.Tensor] = None):
        """
        Z: [B,N,H]  （ブロック表現 at level L）
        S_out: 復元先のシーケンス長（直下の DownScale 入力長）
        token_q: [B,S_out,H]（cross_attn の Query；対応する Down の局所出力を使う）
        """
        B, N, H = Z.shape
        device, dtype = Z.device, Z.dtype
        if self.mode == "overlap_add":
            w = self._kernel(device, dtype)
            A = _build_overlap_matrix(S_out, N, self.W, self.R, device, dtype, w, self.strict_causal)  # [S,N]
            return torch.einsum("sn,bnh->bsh", A, Z)
        else:  # cross_attn
            assert token_q is not None, "cross_attn requires token_q"
            ends = _block_ends(N, self.W, self.R, device)
            return self.cross(token_q, Z, ends, strict_causal=self.strict_causal)


# =========================
# DownScale ユニット
# =========================
class DownUnit(nn.Module):
    """
    入:  [B,S,H]
    出1: [B,S,H]  窓内処理後の局所トークン（overlap-add 縫い合わせ）
    出2: [B,N,H]  窓要約ブロック
    併:  ブロック position_ids（窓終端）とトークン position_ids は上位から受け取る
    """
    def __init__(self, config: PyramidPhi3Config, level_idx: int):
        super().__init__()
        self.H = config.hidden_size
        self.W = config.window_sizes[level_idx]
        self.R = config.window_strides[level_idx]
        self.summary_type = config.summary_type

        self.in_norm = Phi3RMSNorm(self.H, eps=config.rms_norm_eps)
        self.attn = Phi3Attention(config, layer_idx=None)
        self.mlp_norm = Phi3RMSNorm(self.H, eps=config.rms_norm_eps)
        self.mlp = Phi3MLP(config)

        if self.summary_type == "attn":
            self.sum_scorer = nn.Linear(self.H, 1, bias=False)

    def forward(self, x: torch.Tensor, position_ids: torch.LongTensor):
        """
        x: [B,S,H], position_ids: [B,S]
        return:
          y_local:   [B,S,H]  （窓内処理を overlap-add で縫い戻し）
          z_blocks:  [B,N,H]
          block_pos: [B,N]    （各ブロックの終端位置）
        """
        B, S, H = x.shape
        device, dtype = x.device, x.dtype
        W, R = self.W, self.R
        N = (S - W) // R + 1 if S >= W else 1
        starts = torch.arange(N, device=device) * R  # [N]

        # 切り出し
        windows = torch.stack([x[:, s:s+W, :] for s in starts.tolist()], dim=1)             # [B,N,W,H]
        win_pos = torch.stack([position_ids[:, s:s+W] for s in starts.tolist()], dim=1)     # [B,N,W]
        BN = B * N

        # 窓内因果 Self-Attn + MLP
        w_flat = windows.reshape(BN, W, H)
        p_flat = win_pos.reshape(BN, W)
        mask = _lower_tri_mask(W, device, dtype).expand(BN, 1, W, W)
        y = self.in_norm(w_flat)
        y = self.attn(y, attention_mask=mask, position_ids=p_flat, use_cache=False)[0]
        y = w_flat + y
        y2 = self.mlp_norm(y)
        y2 = self.mlp(y2)
        y = y + y2                                 # [BN,W,H]
        y_win = y.view(B, N, W, H)                 # [B,N,W,H]

        # 要約 → ブロック
        if self.summary_type == "attn":
            score = self.sum_scorer(y_win).squeeze(-1)    # [B,N,W]
            alpha = torch.softmax(score, dim=-1)
            z_blocks = torch.einsum("bnw,bnwh->bnh", alpha, y_win)
        else:
            z_blocks = y_win.mean(dim=2)

        # トークンへ縫い戻し（overlap-add）
        blend = _triangular_kernel(W, device, dtype).view(1,1,W,1)
        y_accum = x.new_zeros(B, S, H)
        w_accum = x.new_zeros(B, S, 1)
        for i, s in enumerate(starts.tolist()):
            e = s + W
            y_accum[:, s:e, :] += y_win[:, i, :, :] * blend.squeeze(0).squeeze(0)
            w_accum[:, s:e, :] += blend.squeeze(0).squeeze(0)
        y_local = y_accum / (w_accum + 1e-8)             # [B,S,H]

        # ブロック位置（窓終端）
        block_pos = (starts + (W - 1)).view(1, N).expand(B, N)  # [B,N]
        return y_local, z_blocks, block_pos


# =========================
# UpScale ユニット
# =========================
class UpUnit(nn.Module):
    """
    入:   [B,N,H] （ブロック列）
    操作: ブロック自己注意（因果）→ Upscale（①/③ 切替）
    返:   [B,S_prev,H]  （一段細かい解像度へ）
    """
    def __init__(self, config: PyramidPhi3Config, level_idx: int):
        super().__init__()
        self.H = config.hidden_size
        self.W = config.window_sizes[level_idx]
        self.R = config.window_strides[level_idx]

        self.block_norm = Phi3RMSNorm(self.H, eps=config.rms_norm_eps)
        self.block_attn = Phi3Attention(config, layer_idx=None)
        self.block_mlp_norm = Phi3RMSNorm(self.H, eps=config.rms_norm_eps)
        self.block_mlp = Phi3MLP(config)

        self.up = Upscaler(
            hidden_size=self.H, W=self.W, R=self.R,
            mode=config.up_mode, learnable_kernel=config.learnable_kernel,
            strict_causal=config.strict_causal, num_heads=config.num_attention_heads
        )

    def forward(self, z_blocks: torch.Tensor, block_pos: torch.LongTensor,
                S_prev: int, token_q_prev: Optional[torch.Tensor] = None):
        """
        z_blocks: [B,N,H], block_pos: [B,N]
        S_prev:   復元先の長さ（直前 Down の入力長）
        token_q_prev: [B,S_prev,H]（cross_attn 用の Query。対応する Down で保存した y_local）
        """
        B, N, H = z_blocks.shape
        device, dtype = z_blocks.device, z_blocks.dtype

        # ブロック自己注意（因果）
        z = self.block_norm(z_blocks)
        mask = _lower_tri_mask(N, device, dtype).expand(B, 1, N, N)
        z_out = self.block_attn(z, attention_mask=mask, position_ids=block_pos, use_cache=False)[0]
        z = z_blocks + z_out
        z2 = self.block_mlp_norm(z)
        z2 = self.block_mlp(z2)
        z = z + z2   # [B,N,H]

        # アップスケール
        y_prev = self.up(Z=z, S_out=S_prev, token_q=token_q_prev)  # [B,S_prev,H]
        return y_prev


# =========================
# 1 モデル層 = Down ... Down ... Up ... Up
# =========================
class PyramidPhi3Layer(nn.Module):
    def __init__(self, config: PyramidPhi3Config):
        super().__init__()
        assert config.num_hidden_layers % 2 == 0, "num_hidden_layers must be even per your design"
        self.H = config.hidden_size
        self.n_half = config.num_hidden_layers // 2

        # Down と Up の段を用意
        self.downs = nn.ModuleList([DownUnit(config, i) for i in range(self.n_half)])
        self.ups   = nn.ModuleList([UpUnit(config, i)  for i in reversed(range(self.n_half))])

        # 最後に微調整用の MLP 反復
        self.final_norm = Phi3RMSNorm(self.H, eps=config.rms_norm_eps)
        self.final_mlp  = Phi3MLP(config)
        extra = max(int(config.post_attention_dense_repeats) - 1, 0)
        self.extra_norms = nn.ModuleList([Phi3RMSNorm(self.H, eps=config.rms_norm_eps) for _ in range(extra)])
        self.extra_mlps  = nn.ModuleList([Phi3MLP(config) for _ in range(extra)])

    def forward(self, x: torch.Tensor, position_ids: torch.LongTensor):
        """
        x: [B,S,H], position_ids: [B,S]
        """
        # ---- Down pyramid ----
        # 各段のメタ（復元に必要）をスタックに積む
        down_stack = []  # list of dict{S, y_local, block_pos}
        z = None
        block_pos = None
        for i, down in enumerate(self.downs):
            y_local, z_blocks, block_pos_i = down(x if i == 0 else z,   # 2段目以降はブロック列を「新たなシーケンス」と見なす
                                                  position_ids if i == 0 else block_pos)
            # 次段の入力はブロック列
            down_stack.append({"S": x.size(1) if i == 0 else z.size(1),
                               "y_local": y_local,
                               "block_pos": block_pos_i})
            z, block_pos = z_blocks, block_pos_i

        # ---- Up pyramid（対称に戻す）----
        # 直前 Down から順に取り出して復元
        for j, up in enumerate(self.ups):
            meta = down_stack[-(j+1)]
            S_prev = meta["S"]
            token_q_prev = meta["y_local"]   # cross_attn 用 Query
            x = up(z, block_pos, S_prev, token_q_prev)  # [B,S_prev,H]
            z, block_pos = x, meta["block_pos"]  # 次の Up の「ブロック列/位置」は meta に合わせ直す

        # ---- 最終 MLP（微調整）----
        y = self.final_norm(x)
        y = self.final_mlp(y)
        x = x + y
        for nrm, mlp in zip(self.extra_norms, self.extra_mlps):
            y = nrm(x); y = mlp(y); x = x + y
        return x  # [B,S,H]


# =========================
# Model / LM Head
# =========================
class PyramidPhi3Model(Phi3PreTrainedModel):
    config_class = PyramidPhi3Config
    def __init__(self, config: PyramidPhi3Config):
        super().__init__(config)
        assert config.num_hidden_layers % 2 == 0, "num_hidden_layers must be even"
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layer = PyramidPhi3Layer(config)
        self.norm = Phi3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        if inputs_embeds is None:
            x = self.embed_tokens(input_ids)  # [B,S,H]
        else:
            x = inputs_embeds
        B, S, _ = x.shape
        if position_ids is None:
            position_ids = torch.arange(S, device=x.device).unsqueeze(0).expand(B, S)

        x = self.layer(x, position_ids)
        x = self.norm(x)
        return BaseModelOutputWithPast(last_hidden_state=x)

class PyramidPhi3ForCausalLM(Phi3PreTrainedModel):
    config_class = PyramidPhi3Config
    def __init__(self, config: PyramidPhi3Config):
        super().__init__(config)
        self.model = PyramidPhi3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight
        self.post_init()

    def forward(self, input_ids=None, position_ids=None, labels=None, **kwargs):
        out = self.model(input_ids=input_ids, position_ids=position_ids)
        logits = self.lm_head(out.last_hidden_state)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)),
                                         shift_labels.view(-1))
        return {"loss": loss, "logits": logits, **out.__dict__}
