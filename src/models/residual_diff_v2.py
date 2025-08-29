# coding: utf-8
from typing import Optional, List, Tuple
import torch
import torch.nn.functional as F
from torch import nn

from transformers.models.phi3.configuration_phi3 import Phi3Config
from transformers.models.phi3.modeling_phi3 import (
    Phi3PreTrainedModel,
    Phi3RMSNorm,
    Phi3MLP,
    Phi3SdpaAttention,
    Phi3RotaryEmbedding,
)
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutputWithPast
from transformers.generation.utils import GenerationMixin


# ==============
# helpers (差分/マスク, 形状ユーティリティ, optional RoPE)
# ==============

def first_order_diff(x: torch.Tensor) -> torch.Tensor:
    # (B,L,H) -> (B,L-1,H)
    return x[:, 1:, :] - x[:, :-1, :]

def second_order_diff(x: torch.Tensor) -> torch.Tensor:
    # (B,L,H) -> (B,L-2,H): x_{t+2} - 2 x_{t+1} + x_t
    return x[:, 2:, :] - 2.0 * x[:, 1:-1, :] + x[:, :-2, :]

def mask_and(*tensors: torch.Tensor) -> torch.Tensor:
    # AND を丁寧に（float/bool いずれでも可）
    out = tensors[0].bool()
    for t in tensors[1:]:
        out = out & t.bool()
    return out.to(tensors[0].dtype)

def build_mask_for_diff(mask2d: Optional[torch.Tensor], order: int) -> Optional[torch.Tensor]:
    if mask2d is None:
        return None
    if order == 0:
        return mask2d
    elif order == 1:
        return mask_and(mask2d[:, 1:], mask2d[:, :-1])
    elif order == 2:
        return mask_and(mask2d[:, 2:], mask2d[:, 1:-1], mask2d[:, :-2])
    else:
        raise ValueError("order must be 0,1,2")

def shape_qkv(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    # (B,L,H) -> (B,heads,L,head_dim)
    B, L, H = x.shape
    head_dim = H // num_heads
    x = x.view(B, L, num_heads, head_dim).transpose(1, 2)  # (B,heads,L,head_dim)
    return x

def unshape_ctx(x: torch.Tensor) -> torch.Tensor:
    # (B,heads,L,head_dim) -> (B,L,H)
    B, nH, L, d = x.shape
    return x.transpose(1, 2).contiguous().view(B, L, nH * d)

# RoPE helper（必要な場合のみ使用）
def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = torch.cat([(q_rot * cos) + (rotate_half(q_rot) * sin), q_pass], dim=-1)
    k_embed = torch.cat([(k_rot * cos) + (rotate_half(k_rot) * sin), k_pass], dim=-1)
    return q_embed, k_embed


# ==============
# Self-Block （Phi-3 部品そのまま）
# ==============

class Phi3SelfBlock(nn.Module):
    """
    PreNorm -> Self-Attn(SDPA+RoPE) -> resid -> PreNorm -> MLP -> resid
    """
    def __init__(self, config: Phi3Config, layer_idx: int, rotary_emb: Phi3RotaryEmbedding):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.in_norm = Phi3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = Phi3SdpaAttention(config, layer_idx=layer_idx)
        self.dropout_attn = nn.Dropout(config.resid_pdrop)
        self.ff_norm = Phi3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = Phi3MLP(config)
        self.dropout_mlp = nn.Dropout(config.resid_pdrop)
        self.rotary_emb = rotary_emb

    def _prepare_4d_mask(self, mask2d, bsz, seqlen, hidden_states):
        if mask2d is None:
            return None
        return _prepare_4d_causal_attention_mask(
            mask2d, (bsz, seqlen), hidden_states, past_key_values_length=0, sliding_window=self.config.sliding_window
        )

    def forward(
        self,
        hidden_states: torch.Tensor,         # (B,L,H)
        mask2d: Optional[torch.Tensor],      # (B,L)
        position_ids: Optional[torch.Tensor],# unused here; we re-make per len
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.in_norm(hidden_states)
        B, L, _ = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        attn_mask = self._prepare_4d_mask(mask2d, B, L, x)

        attn_out, attn_weights, _ = self.attn(
            hidden_states=x,
            attention_mask=attn_mask,
            position_ids=pos,
            rotary_emb=self.rotary_emb,
            past_key_value=None,
            output_attentions=output_attentions,
            use_cache=False,
        )
        x = hidden_states + self.dropout_attn(attn_out)

        h = self.ff_norm(x)
        h = self.mlp(h)
        x = x + self.dropout_mlp(h)
        return x, (attn_weights if output_attentions else None)


# ==============
# Cross-Attention （シンプル実装／GQA対応。既定で RoPE 無し）
# ==============

class SimpleCrossAttention(nn.Module):
    """
    Query: x_q (B,Lq,H)   Key/Value: x_kv (B,Lk,H)
    - num_heads / num_key_value_heads は Phi-3 と同じ設定に合わせる（GQA対応）
    - 既定: RoPE 適用なし（decoder-encoder cross は相対位置の意味付けが曖昧なため）。
      use_rope_in_cross_attn=True で RoPE を適用可能。
    """
    def __init__(self, config: Phi3Config, rotary_emb: Phi3RotaryEmbedding, use_rope_in_cross_attn: bool = False):
        super().__init__()
        self.config = config
        self.rotary_emb = rotary_emb
        self.use_rope = use_rope_in_cross_attn

        H = config.hidden_size
        nH = config.num_attention_heads
        nKV = getattr(config, "num_key_value_heads", nH)
        self.nH = nH
        self.nKV = nKV
        self.groups = nH // nKV
        self.head_dim = H // nH

        self.q_proj = nn.Linear(H, H, bias=False)
        self.k_proj = nn.Linear(H, nKV * self.head_dim, bias=False)
        self.v_proj = nn.Linear(H, nKV * self.head_dim, bias=False)
        self.o_proj = nn.Linear(H, H, bias=False)
        self.dropout = nn.Dropout(config.attn_pdrop)

        # 追加正規化（安定のため）
        self.q_norm = Phi3RMSNorm(H, eps=config.rms_norm_eps)
        self.kv_norm = Phi3RMSNorm(H, eps=config.rms_norm_eps)

    def _kv_repeat(self, x: torch.Tensor) -> torch.Tensor:
        # (B, nKV, Lk, d) -> (B, nH, Lk, d) へ繰り返し
        if self.nKV == self.nH:
            return x
        return x.repeat_interleave(self.groups, dim=1)

    def _make_attn_mask_bool(self, enc_mask2d: Optional[torch.Tensor], Lq: int, Lk: int, B: int) -> Optional[torch.Tensor]:
        # enc_mask2d: (B,Lk) in {0,1}  -> broadcastable bool mask of shape (B,1,Lq,Lk)
        if enc_mask2d is None:
            return None
        m = (~enc_mask2d.bool()).unsqueeze(1).unsqueeze(2)  # True=mask
        return m.expand(B, 1, Lq, Lk)

    def forward(
        self,
        x_q: torch.Tensor,             # (B,Lq,H)
        x_kv: torch.Tensor,            # (B,Lk,H)
        enc_mask2d: Optional[torch.Tensor] = None,  # (B,Lk)
    ) -> torch.Tensor:
        B, Lq, H = x_q.shape
        Lk = x_kv.size(1)

        q = self.q_proj(self.q_norm(x_q))                      # (B,Lq,H)
        k = self.k_proj(self.kv_norm(x_kv))                    # (B,Lk, nKV*Hd)
        v = self.v_proj(self.kv_norm(x_kv))                    # (B,Lk, nKV*Hd)

        q = shape_qkv(q, self.nH)                              # (B,nH,Lq,Hd)
        k = k.view(B, Lk, self.nKV, self.head_dim).transpose(1, 2)  # (B,nKV,Lk,Hd)
        v = v.view(B, Lk, self.nKV, self.head_dim).transpose(1, 2)  # (B,nKV,Lk,Hd)
        k = self._kv_repeat(k)                                 # (B,nH,Lk,Hd)
        v = self._kv_repeat(v)                                 # (B,nH,Lk,Hd)

        if self.use_rope:
            # 参考実装：各系列長に合わせて cos/sin を取り、q/k に適用
            # Phi3RotaryEmbedding の forward は (x, position_ids) -> (cos, sin) を返す前提
            pos_q = torch.arange(Lq, device=x_q.device).unsqueeze(0).expand(B, -1)
            pos_k = torch.arange(Lk, device=x_q.device).unsqueeze(0).expand(B, -1)
            # ダミーの [B,L,head_dim] を渡して cos/sin を得る（実装に依存するため try/except）
            try:
                dummy_q = torch.zeros(B, Lq, self.head_dim, device=x_q.device, dtype=x_q.dtype)
                dummy_k = torch.zeros(B, Lk, self.head_dim, device=x_q.device, dtype=x_q.dtype)
                cos_q, sin_q = self.rotary_emb(dummy_q, pos_q)
                cos_k, sin_k = self.rotary_emb(dummy_k, pos_k)
                q, k = apply_rotary_pos_emb(q, k, cos_q, sin_k, position_ids=None, unsqueeze_dim=2)
            except Exception:
                # RoPE 未対応環境では静かにスキップ（Self-Attn 側で RoPE が効いていれば全体としては相対位置信号を保持）
                pass

        # scaled dot-product attention
        attn_mask = self._make_attn_mask_bool(enc_mask2d, Lq, Lk, B)  # True=mask
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout.p if self.training else 0.0, is_causal=False
        )  # (B,nH,Lq,Hd)

        y = unshape_ctx(y)                                           # (B,Lq,H)
        y = self.o_proj(y)
        return y


# ==============
# v2: 3本の枝を「各3層」通してから、0階に Cross-Attn(←1階) → Cross-Attn(←2階)
# ==============

class DiffUpscalePhi3ModelV2(Phi3PreTrainedModel):
    """
    1) embedding -> 0/1/2階差分 3枝
    2) 各枝: [SelfBlock] x 3 （同一枝内で3層）
    3) x0 に CrossAttn(x1_final) → residual、続けて CrossAttn(x2_final) → residual
    出力は x0（原系列長 L）
    """
    def __init__(self, config: Phi3Config, use_rope_in_cross_attn: bool = False):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.rotary_emb = Phi3RotaryEmbedding(config=config)
        self.norm_out = Phi3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 各枝 3 層
        self.branch0 = nn.ModuleList([Phi3SelfBlock(config, layer_idx=i, rotary_emb=self.rotary_emb) for i in range(3)])
        self.branch1 = nn.ModuleList([Phi3SelfBlock(config, layer_idx=100+i, rotary_emb=self.rotary_emb) for i in range(3)])
        self.branch2 = nn.ModuleList([Phi3SelfBlock(config, layer_idx=200+i, rotary_emb=self.rotary_emb) for i in range(3)])

        # Cross-Attn × 2 （0階 <- 1階, 0階 <- 2階）
        self.cross01_norm = Phi3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.cross01 = SimpleCrossAttention(config, self.rotary_emb, use_rope_in_cross_attn)
        self.cross12_norm = Phi3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.cross02 = SimpleCrossAttention(config, self.rotary_emb, use_rope_in_cross_attn)

        self.dropout = nn.Dropout(config.resid_pdrop)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,   # (B,L)
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> BaseModelOutput:
        output_attentions = False if output_attentions is None else output_attentions
        output_hidden_states = False if output_hidden_states is None else output_hidden_states
        return_dict = True if return_dict is None else return_dict

        if inputs_embeds is None:
            x0 = self.embed_tokens(input_ids)  # (B,L,H)  0階
        else:
            x0 = inputs_embeds
        B, L, H = x0.shape

        m0 = attention_mask if attention_mask is not None else torch.ones(B, L, device=x0.device, dtype=torch.long)
        # 1階/2階差分
        x1 = first_order_diff(x0)      # (B,L-1,H)
        x2 = second_order_diff(x0)     # (B,L-2,H)
        m1 = build_mask_for_diff(m0, 1)
        m2 = build_mask_for_diff(m0, 2)

        # 各枝 3 層
        for blk in self.branch0:
            x0, _ = blk(x0, m0, None, output_attentions=False)
        for blk in self.branch1:
            x1, _ = blk(x1, m1, None, output_attentions=False)
        for blk in self.branch2:
            x2, _ = blk(x2, m2, None, output_attentions=False)

        # CrossAttn: x0 <- x1
        x0 = x0 + self.dropout(self.cross01(self.cross01_norm(x0), x1, enc_mask2d=m1))
        # CrossAttn: x0 <- x2
        x0 = x0 + self.dropout(self.cross02(self.cross12_norm(x0), x2, enc_mask2d=m2))

        x_out = self.norm_out(x0)

        if not return_dict:
            return (x_out,)

        return BaseModelOutput(
            last_hidden_state=x_out,
            hidden_states=None,
            attentions=None,
        )


class DiffUpscalePhi3ForCausalLMV2(Phi3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Phi3Config, use_rope_in_cross_attn: bool = False):
        super().__init__(config)
        self.model = DiffUpscalePhi3ModelV2(config, use_rope_in_cross_attn=use_rope_in_cross_attn)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # weight tying
        self.lm_head.weight = self.model.embed_tokens.weight
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=kwargs.get("output_attentions", False),
            output_hidden_states=kwargs.get("output_hidden_states", False),
            return_dict=True,
        )
        logits = self.lm_head(out.last_hidden_state).float()

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))

        return CausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=None, hidden_states=None, attentions=None
        )

    @property
    def base_model(self):
        return self.model


# ==============
# v3: 「各枝1層 + x0<-x1 Cross + x0<-x2 Cross」を1ブロックとして **3回** 反復
# ==============

class DiffUpscalePhi3ModelV3(Phi3PreTrainedModel):
    """
    1 block = { 3枝: SelfBlock各1層 → x0<-x1 Cross → x0<-x2 Cross }
    これを 3 回繰り返す（早期融合 + 反復洗練）
    """
    def __init__(self, config: Phi3Config, use_rope_in_cross_attn: bool = False):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.rotary_emb = Phi3RotaryEmbedding(config=config)
        self.norm_out = Phi3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.dropout = nn.Dropout(config.resid_pdrop)

        # 3 ブロック分の層を用意（枝それぞれ + CrossAttn×2）
        self.blocks_branch0 = nn.ModuleList([Phi3SelfBlock(config, layer_idx=10+i, rotary_emb=self.rotary_emb) for i in range(3)])
        self.blocks_branch1 = nn.ModuleList([Phi3SelfBlock(config, layer_idx=110+i, rotary_emb=self.rotary_emb) for i in range(3)])
        self.blocks_branch2 = nn.ModuleList([Phi3SelfBlock(config, layer_idx=210+i, rotary_emb=self.rotary_emb) for i in range(3)])

        self.cross_norm_01 = nn.ModuleList([Phi3RMSNorm(config.hidden_size, eps=config.rms_norm_eps) for _ in range(3)])
        self.cross_01 = nn.ModuleList([SimpleCrossAttention(config, self.rotary_emb, use_rope_in_cross_attn) for _ in range(3)])

        self.cross_norm_02 = nn.ModuleList([Phi3RMSNorm(config.hidden_size, eps=config.rms_norm_eps) for _ in range(3)])
        self.cross_02 = nn.ModuleList([SimpleCrossAttention(config, self.rotary_emb, use_rope_in_cross_attn) for _ in range(3)])

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> BaseModelOutput:
        output_attentions = False if output_attentions is None else output_attentions
        output_hidden_states = False if output_hidden_states is None else output_hidden_states
        return_dict = True if return_dict is None else return_dict

        if inputs_embeds is None:
            x0 = self.embed_tokens(input_ids)
        else:
            x0 = inputs_embeds
        B, L, H = x0.shape
        m0 = attention_mask if attention_mask is not None else torch.ones(B, L, device=x0.device, dtype=torch.long)

        # 初回の差分（1,2階）は x0 から
        def mk_x1x2(x0, m0):
            return first_order_diff(x0), second_order_diff(x0), build_mask_for_diff(m0, 1), build_mask_for_diff(m0, 2)

        x1, x2, m1, m2 = mk_x1x2(x0, m0)

        # 3 ブロック反復
        for i in range(3):
            # 各枝 1 層
            x0, _ = self.blocks_branch0[i](x0, m0, None, output_attentions=False)
            x1, _ = self.blocks_branch1[i](x1, m1, None, output_attentions=False)
            x2, _ = self.blocks_branch2[i](x2, m2, None, output_attentions=False)

            # Cross: x0 <- x1, ついで x0 <- x2
            x0 = x0 + self.dropout(self.cross_01[i](self.cross_norm_01[i](x0), x1, enc_mask2d=m1))
            x0 = x0 + self.dropout(self.cross_02[i](self.cross_norm_02[i](x0), x2, enc_mask2d=m2))

            # 次ブロック用に 1/2階差分を「最新の x0」から再計算する手もある
            # （Down(2) 的な早期融合→再分解の設計に合わせたい場合）
            # ここでは「枝連鎖の継続」を優先し、x1/x2 は枝内の連続層として進める。
            # もし再分解を望むなら下記を有効化：
            # x1, x2, m1, m2 = mk_x1x2(x0, m0)

        x_out = self.norm_out(x0)

        if not return_dict:
            return (x_out,)

        return BaseModelOutput(
            last_hidden_state=x_out,
            hidden_states=None,
            attentions=None,
        )


class DiffUpscalePhi3ForCausalLMV3(Phi3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Phi3Config, use_rope_in_cross_attn: bool = False):
        super().__init__(config)
        self.model = DiffUpscalePhi3ModelV3(config, use_rope_in_cross_attn=use_rope_in_cross_attn)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=kwargs.get("output_attentions", False),
            output_hidden_states=kwargs.get("output_hidden_states", False),
            return_dict=True,
        )
        logits = self.lm_head(out.last_hidden_state).float()

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))

        return CausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=None, hidden_states=None, attentions=None
        )

    @property
    def base_model(self):
        return self.model
