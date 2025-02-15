# memo
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/phi3/modeling_phi3.py#L257
# このあたりを修正

import torch
import torch.nn as nn

from transformers import Phi3Decoder, Phi3Config, Phi3Attention
from types import Optional, Tuple

from transformers.processing_utils import Unpack
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs


class Router(nn.Module):
    def __init__(self, embedding_dim, parallel_num):
        """
        Args:
            embedding_dim:
            parallel_num:
        """
        super(Router, self).__init__()
        self.gate = nn.Linear(embedding_dim, parallel_num)
        self.parallel_num = parallel_num

    def forward(self, hidden_states):
        """
        Args:
            hidden_states (`torch.FloatTensor`):
        """
        hidden_states = self.gate(hidden_states)
        hidden_states = torch.mean(hidden_states, dim=1)
        hidden_states = torch.sigmoid(hidden_states)
        # (batch_size, n) -> (batch_size, n) with values of 0 or 1
        hidden_states = torch.where(hidden_states > 0.5, 1, 0)

        hidden_states = torch.mean(hidden_states, dim=1)
        # (batch, embed_dim) -> (batch, N)
        hidden_states = self.gate(hidden_states)
        # (batch, N) -> (batch, N)
        hidden_states = torch.softmax(hidden_states, dim=1)
        # (batch, N) -> (batch, N)
        hidden_states = torch.argmax(hidden_states, dim=1)  # 最大値のインデックスを取得
        # (batch, N) -> (batch, N)
        hidden_states = torch.nn.functional.one_hot(
            hidden_states, num_classes=self.parallel_num
        )  # one-hotエンコーディング

        return hidden_states


class MeshDecoder(Phi3Decoder):
    def __init__(self, config: Phi3Config, layer_idx: int):
        parallel_num = 3
        self.router = Router(self.hidden_size, parallel_num)

        self.self_attn_layers = nn.ModuleList(
            [
                Phi3Attention(config=config, layer_idx=layer_idx)
                for _ in range(parallel_num)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range
                `[0, config.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
            past_key_value (`Cache`, *optional*): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        # (batch, seq_len, embed_dim) -> (batch, N)
        route = self.router(hidden_states)
        # (batch, N) -> (batch, 1)
        indices = torch.argmax(route, dim=1).unsqueeze(1)

        _hidden_states = []
        _self_attn_weights = []
        for i in range(hidden_states.size(0)):
            layer_index = indices[i]
            attn = self.self_attn_layers[layer_index]
            hidden_states, self_attn_weights = attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            _hidden_states.append(hidden_states)
            _self_attn_weights.append(self_attn_weights)

        hidden_states = torch.stack(_hidden_states)
        self_attn_weights = torch.stack(_self_attn_weights)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        return outputs
