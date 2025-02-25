
import torch
from torch import nn

from typing import Callable, List, Optional, Tuple, Union
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.generation import GenerationMixin

from .phi3 import (
    Phi3DecoderLayer,
    Phi3Model,
    Phi3ForCausalLM,
    Phi3PreTrainedModel,

)
from .phi3_config import Phi3Config


class FewAttentionConfig(Phi3Config):
    def __init__(self, skip_index = [], *args, **kwargs):
        super().__init(*args, **kwargs)
        self.skip_index = skip_index

class SkipedDecoderLayer(Phi3DecoderLayer):
    def __init__(self, config: FewAttentionConfig, layer_idx: int, skip_attention = False):
        super().__init__(config, layer_idx)
        self.skip_attention = skip_attention

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
        if self.skip_attention:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + self.resid_mlp_dropout(
                hidden_states
            )
            return hidden_states
        else:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            # Self Attention
            hidden_states, self_attn_weights = self.self_attn(
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
            hidden_states = residual + self.resid_attn_dropout(
                hidden_states
            )  # main diff with Llama

            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)

            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + self.resid_mlp_dropout(
                hidden_states
            )  # main diff with Llama

            outputs = (hidden_states,)
            if output_attentions:
                outputs += (self_attn_weights,)

            return outputs


class FewAttentionModel(Phi3Model):
    def __init__(self, config: FewAttentionConfig):
        super().__init__(config)
        skip_index = [1, 2, 3]
        self.layers = nn.ModuleList(
            [
                SkipedDecoderLayer(config, layer_idx, skip_attention=layer_idx in skip_index)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

class FewAttentionModelForCausalLM(Phi3PreTrainedModel, GenerationMixin):
    def __init__(self, config: FewAttentionConfig):
        super().__init__(config)
        self.model = FewAttentionModel(config)

