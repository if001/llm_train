"""
「差分→Attn→Dense を前半層で繰り返し、後半層は“Denseでのアップスケール(=長さ+1)”→Attn→Dense を繰り返して最終的に元の seq_len に戻す」アーキテクチャ
"""

from typing import Optional, Tuple, List
import torch
from torch import nn

from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerationMixin, GenerateDecoderOnlyOutput

# from transformers.models.phi3.configuration_phi3 import Phi3Config
# from transformers.models.phi3.modeling_phi3 import (
#     Phi3PreTrainedModel,
#     Phi3RotaryEmbedding,
#     Phi3RMSNorm,
#     Phi3Attention,
#     # Phi3SdpaAttention,   # 既定の SDPA 注意
#     Phi3MLP,
# )
from models.phi3_config import Phi3Config
from models.phi3 import (
    Phi3PreTrainedModel,
    Phi3RMSNorm,
    Phi3MLP,
    # Phi3SdpaAttention,
    Phi3Attention,
    Phi3RotaryEmbedding,
)


class ResidualNetConfig(Phi3Config):
    model_type = "residualnet"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.tie_word_embeddings = True


# ---------- 長さ変換用の前処理 ----------
class DiffPreprocessor(nn.Module):
    """一次差分: (B, L, H) -> (B, L-1, H) と 2D mask の AND 縮約"""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask_2d: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # hidden_states: (B, L, H)
        x1 = hidden_states[:, 1:, :]
        x0 = hidden_states[:, :-1, :]
        diff = x1 - x0  # (B, L-1, H)

        if attention_mask_2d is not None:
            m = (attention_mask_2d[:, 1:].bool() & attention_mask_2d[:, :-1].bool()).to(
                attention_mask_2d.dtype
            )
        else:
            m = None
        return diff, m


class IntegratePreprocessor(nn.Module):
    """
    学習可能な“積分”で (B, m, H) -> (B, m+1, H)
      1) seed y0 = MLP(mean_pool(z))
      2) y = cumsum([y0, z], dim=1)
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.seed_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask_2d: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # hidden_states: (B, m, H)
        if attention_mask_2d is not None:
            denom = attention_mask_2d.sum(dim=1, keepdim=True).clamp_min(1)
            pooled = (hidden_states * attention_mask_2d.unsqueeze(-1)).sum(
                dim=1
            ) / denom  # (B, H)
            batch_valid = (attention_mask_2d.sum(dim=1) > 0).to(
                attention_mask_2d.dtype
            )  # (B,)
        else:
            pooled = hidden_states.mean(dim=1)
            batch_valid = None

        y0 = self.seed_mlp(pooled).unsqueeze(1)  # (B,1,H)
        y = torch.cumsum(torch.cat([y0, hidden_states], dim=1), dim=1)  # (B, m+1, H)

        if attention_mask_2d is not None:
            new_first = batch_valid.unsqueeze(1)  # (B,1)
            mask = torch.cat([new_first, attention_mask_2d], dim=1)
        else:
            mask = None
        return y, mask


# ---------- レイヤーブロック（Phi3 部品で構成） ----------


class ResidualDiffLayer(nn.Module):
    """
    (差分で L-1) -> Attn -> MLP
    - RoPE は Phi-3 と同様に Attention 内で適用
    - 各層で position_ids を 0..len-1 に張り直す
    """

    def __init__(
            self, config: ResidualNetConfig, layer_idx: int, rotary_emb: Phi3RotaryEmbedding, enable=True,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.input_norm = Phi3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre = DiffPreprocessor()
        # self.attn = Phi3SdpaAttention(config, layer_idx=layer_idx)
        self.attn = Phi3Attention(config, layer_idx=layer_idx)
        self.dropout_attn = nn.Dropout(config.resid_pdrop)
        self.post_norm = Phi3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = Phi3MLP(config)
        self.dropout_mlp = nn.Dropout(config.resid_pdrop)
        self.rotary_emb = rotary_emb  # 共有 RoPE インスタンス
        self.enable = enable
        
    def _to_4d_mask(
        self,
        mask2d: Optional[torch.Tensor],
        bsz: int,
        seqlen: int,
        hidden_states: torch.Tensor,
        past_kv_len: int,
    ) -> Optional[torch.Tensor]:
        if mask2d is None:
            return None
        else:
            if past_kv_len > 0:
                past = torch.ones(
                    (bsz, past_kv_len),
                    dtype=mask2d.dtype,
                    device=mask2d.device,
                )
                attn_2d = torch.cat([past, mask2d], dim=1)  # [B, past_kv_len + seqlen]
            else:
                attn_2d = mask2d
        return _prepare_4d_causal_attention_mask(
            attn_2d,
            (bsz, seqlen),
            hidden_states,
            past_key_values_length=past_kv_len,
            sliding_window=self.config.sliding_window,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # (B, L, H)
        attention_mask_2d: Optional[torch.Tensor],  # (B, L)
        position_ids: Optional[torch.LongTensor],  # (B, L)
        output_attentions: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,  # (B, L)
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        x = self.input_norm(hidden_states)

        if x.size(1) == 1 and self.enable == True:
            # bsz, seqlen, _ = x.shape
            # mask2d = attention_mask_2d
            raise ValueError("seq len must set > 1")

        if self.enable:
            x, mask2d = self.pre(x, attention_mask_2d)  # L→L-1
            bsz, seqlen, _ = x.shape
        else:
            bsz, seqlen, _ = x.shape
            mask2d = attention_mask_2d

        device = x.device

        if cache_position is not None:
            end_pos = cache_position[-1]  # [B]
            # 各バッチ同一想定（現実装は batch=1 前提）。batch>1でも一貫させるなら gather 等で個別生成も可
            end_val = int(end_pos.item())
            start_val = end_val - (seqlen - 1)
            pos_ids = (
                torch.arange(start_val, end_val + 1, device=device)
                .unsqueeze(0)
                .expand(bsz, -1)
            )  # [B,seqlen]
        else:
            # 生成外（学習時など）。past_kv_len を起点に絶対位置を張る
            past_kv_len = 0
            pos_ids = (
                torch.arange(past_kv_len, past_kv_len + seqlen, device=device)
                .unsqueeze(0)
                .expand(bsz, -1)
            )

        position_embeddings = self.rotary_emb(hidden_states, pos_ids)
        past_kv_len = (
            int(cache_position[0].item()) if (cache_position is not None) else 0
        )
        attn_mask_4d = self._to_4d_mask(mask2d, bsz, seqlen, x, past_kv_len)

        attn_out, attn_weights = self.attn(
            hidden_states=x,
            attention_mask=attn_mask_4d,
            position_ids=pos_ids,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value if use_cache else None,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )
        x = x + self.dropout_attn(attn_out)
        h = self.post_norm(x)
        h = self.mlp(h)
        x = x + self.dropout_mlp(h)
        return x, mask2d, attn_weights if output_attentions else None


class IntegrateUpscaleLayer(nn.Module):
    """
    (積分で L+1) -> Attn -> MLP
    """

    def __init__(
            self, config: ResidualNetConfig, layer_idx: int, rotary_emb: Phi3RotaryEmbedding, enable=True
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.input_norm = Phi3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre = IntegratePreprocessor(config.hidden_size)
        # self.attn = Phi3SdpaAttention(config, layer_idx=layer_idx)
        self.attn = Phi3Attention(config, layer_idx=layer_idx)
        self.dropout_attn = nn.Dropout(config.resid_pdrop)
        self.post_norm = Phi3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = Phi3MLP(config)
        self.dropout_mlp = nn.Dropout(config.resid_pdrop)
        self.rotary_emb = rotary_emb
        self.enable = enable

    def _to_4d_mask(
        self,
        mask2d: Optional[torch.Tensor],
        bsz: int,
        seqlen: int,
        hidden_states: torch.Tensor,
        past_kv_len: int,
    ) -> Optional[torch.Tensor]:
        if mask2d is None:
            return None
        else:
            if past_kv_len > 0:
                past = torch.ones(
                    (bsz, past_kv_len),
                    dtype=mask2d.dtype,
                    device=mask2d.device,
                )
                attn_2d = torch.cat([past, mask2d], dim=1)  # [B, past_kv_len + seqlen]
            else:
                attn_2d = mask2d
        return _prepare_4d_causal_attention_mask(
            attn_2d,
            (bsz, seqlen),
            hidden_states,
            past_key_values_length=past_kv_len,
            sliding_window=self.config.sliding_window,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # (B, L, H)
        attention_mask_2d: Optional[torch.Tensor],  # (B, L)
        position_ids: Optional[torch.LongTensor],  # (B, L)
        output_attentions: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        x = self.input_norm(hidden_states)

        if self.enable:
            x, mask2d = self.pre(x, attention_mask_2d)  # L→L+1
            bsz, seqlen, _ = x.shape
        else:
            bsz, seqlen, _ = x.shape
            mask2d = attention_mask_2d

        device = x.device
        if cache_position is not None:
            end_pos = cache_position[-1]
            end_val = int(end_pos.item())
            start_val = end_val - (seqlen - 1)
            pos_ids = (
                torch.arange(start_val, end_val + 1, device=device)
                .unsqueeze(0)
                .expand(bsz, -1)
            )
        else:
            past_kv_len = 0
            pos_ids = (
                torch.arange(past_kv_len, past_kv_len + seqlen, device=device)
                .unsqueeze(0)
                .expand(bsz, -1)
            )

        position_embeddings = self.rotary_emb(hidden_states, pos_ids)

        past_kv_len = (
            int(cache_position[0].item()) if (cache_position is not None) else 0
        )
        attn_mask_4d = self._to_4d_mask(mask2d, bsz, seqlen, x, past_kv_len)

        attn_out, attn_weights = self.attn(
            hidden_states=x,
            attention_mask=attn_mask_4d,
            position_ids=pos_ids,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value if use_cache else None,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )
        x = x + self.dropout_attn(attn_out)
        h = self.post_norm(x)
        h = self.mlp(h)
        x = x + self.dropout_mlp(h)
        return x, mask2d, attn_weights if output_attentions else None


# ---------- モデル本体（Phi3PreTrainedModel を継承） ----------


class ResidualNetModel(Phi3PreTrainedModel):
    """
    前半: ResidualDiffLayer × (N/2) で系列長を縮約
    後半: IntegrateUpscaleLayer × (N/2) で系列長を復元
    """

    config_class = ResidualNetConfig

    def __init__(self, config: ResidualNetConfig):
        super().__init__(config)
        assert (
            config.num_hidden_layers % 2 == 0
        ), "num_hidden_layers は偶数にしてください。"
        assert (
            config.num_hidden_layers % 4 == 0
        ), "num_hidden_layers は4の倍数にしてください"

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.norm = Phi3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Phi3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False


        if config.num_hidden_layers <= 4:
            half = config.num_hidden_layers // 2
            # 前半 (down)
            self.down_layers = nn.ModuleList(
                [
                    ResidualDiffLayer(config, layer_idx=i, rotary_emb=self.rotary_emb)
                    for i in range(half)
                ]
            )
            # 後半 (up)
            self.up_layers = nn.ModuleList(
                [
                    IntegrateUpscaleLayer(
                        config, layer_idx=half + i, rotary_emb=self.rotary_emb
                    )
                    for i in range(half)
                ]
            )
        else:
            _downs = []
            _part = config.num_hidden_layers // 4
            for i in range(_part):
                _downs.append(ResidualDiffLayer(config, layer_idx=i, rotary_emb=self.rotary_emb))
                _downs.append(ResidualDiffLayer(config, layer_idx=i+1, rotary_emb=self.rotary_emb, enable=False))
            self.down_layers = nn.ModuleList(_downs)
            
            _up = []
            for i in range(_part):
                _up.append(IntegrateUpscaleLayer(config, layer_idx=i+_part, rotary_emb=self.rotary_emb))
                _up.append(IntegrateUpscaleLayer(config, layer_idx=i+_part+1, rotary_emb=self.rotary_emb, enable=False))
            self.up_layers = nn.ModuleList(_up)
                

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,  # (B, L) in {0,1}
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_cache: bool = False,
        past_key_values: Optional[List[torch.Tensor]] = None,  # List[LayerKV] or None
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        output_attentions = (
            output_attentions if output_attentions is not None else False
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else False
        )
        return_dict = True if return_dict is None else return_dict

        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must specify either input_ids or inputs_embeds.")

        if inputs_embeds is None:
            hidden_states = self.embed_tokens(input_ids)  # (B, L, H)
        else:
            hidden_states = inputs_embeds

        mask2d = attention_mask
        bsz, orig_len, _ = hidden_states.shape

        all_hidden_states: List[torch.Tensor] = [] if output_hidden_states else None
        all_attns: List[torch.Tensor] = [] if output_attentions else None

        # ---- 前半: 差分で縮約 ----
        for layer in self.down_layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            hidden_states, mask2d, attn = layer(
                hidden_states,
                mask2d,
                position_ids,
                output_attentions=output_attentions,
                past_key_value=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
            )
            if output_attentions:
                all_attns.append(attn)

        # ---- 後半: 積分で復元 ----
        for layer in self.up_layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            hidden_states, mask2d, attn = layer(
                hidden_states,
                mask2d,
                position_ids,
                output_attentions=output_attentions,
                past_key_value=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
            )
            if output_attentions:
                all_attns.append(attn)

        # 最終長の整合性（念のため）
        if hidden_states.size(1) != orig_len:
            raise RuntimeError(
                f"seq_len が復元されていません: got {hidden_states.size(1)} vs {orig_len}"
            )

        hidden_states = self.norm(hidden_states)

        if not return_dict:
            out = (hidden_states,)
            if output_hidden_states:
                out = out + (all_hidden_states,)
            if output_attentions:
                out = out + (all_attns,)
            return out

        return {
            "last_hidden_state": hidden_states,
            "hidden_states": all_hidden_states,
            "attentions": all_attns,
        }


# ---------- CausalLM ヘッド（Phi3PreTrainedModel + GenerationMixin） ----------


class ResidualNetForCausalLM(Phi3PreTrainedModel, GenerationMixin):
    config_class = ResidualNetConfig
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config: ResidualNetConfig):
        super().__init__(config)
        self.model = ResidualNetModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # weight tying
        # self.lm_head.weight = self.model.embed_tokens.weight

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_cache: bool = False,
        past_key_values: Optional[List[torch.Tensor]] = None,  # List[LayerKV] or None
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        return_dict = True if return_dict is None else return_dict

        model_out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            use_cache=use_cache,
            past_key_value=past_key_values if use_cache else None,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = model_out["last_hidden_state"]  # (B, L, H)
        logits = self.lm_head(hidden_states).float()

        loss = None
        if labels is not None:
            # 因果言語モデリング損失
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.vocab_size), shift_labels.view(-1)
            )

        if not return_dict:
            return (logits, loss)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=model_out["hidden_states"],
            attentions=model_out["attentions"],
        )

    @property
    def base_model(self):
        return self.model

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional["Cache"] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ):
        """
        - prefill（past_key_values None）では全文をそのまま渡し、cache_position = [0..L-1]
        - decode（past_key_values あり）では直近3トークンだけを入力し、
          cache_position = [keep_len .. keep_len+Lw-1] とする
            * t = 直前に確定した末尾 index（= input_ids.size(1)-1）
            * keep_len = max(0, t-2)
        - attention_mask は簡単のため 1 埋め（左PAD運用ならそのまま attention_mask を流用）
        """
        device = input_ids.device
        use_cache = True if use_cache is None else use_cache

        if past_key_values is None or not use_cache:
            # ---- prefill: 全文 ----
            L = input_ids.size(1)
            cache_position = torch.arange(0, L, device=device)        # [L]
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, device=device)
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "cache_position": cache_position,                     # ★ [L]
            }

        # ---- decode: 3トークン窓 ----
        # input_ids は「これまでの全文」なので、ここで直近3に切る
        window = input_ids[:, -3:]                                     # [B, Lw<=3]
        Lw = window.size(1)
        t = input_ids.size(1) - 1                                      # i, i+1, ...
        keep_len = max(0, t - 2)                                       # 仕様: “2つ前まで”を過去KVとして可視
        cache_position = torch.arange(keep_len, keep_len + Lw, device=device)  # ★ [Lw]

        # attention_mask は窓に合わせる（左PADを使っているなら適宜スライスに置換）
        attn = torch.ones_like(window, device=device)

        return {
            "input_ids": window,
            "attention_mask": attn,
            "past_key_values": past_key_values,                         # Cache（in-place更新）
            "use_cache": use_cache,
            "cache_position": cache_position,                           # ★ [Lw]
        }
    
