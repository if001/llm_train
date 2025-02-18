from typing import Callable, List, Optional, Tuple, Union

import torch
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import (
    Cache,
    DynamicCache,
    SlidingWindowCache,
    StaticCache,
)
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.utils import (
    LossKwargs,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

# from configuration_phi3 import Phi3Config  # Assuming configuration is in the same directory
from models.phi3 import (
    Phi3ForCausalLM,
    Phi3MLP,
    Phi3Attention,
    Phi3RMSNorm,
    Phi3RotaryEmbedding,
    Phi3Config,
    Phi3Model,
    Phi3PreTrainedModel,
    PHI3_INPUTS_DOCSTRING,
    _CONFIG_FOR_DOC,
)

logger = logging.get_logger(__name__)

import torch
from torch import nn
import torch.nn.functional as F
from transformers.cache_utils import (
    Cache,
    DynamicCache,
    SlidingWindowCache,
    StaticCache,
)
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from typing import Optional, Tuple, Union
from .phi3 import Phi3Config, Phi3RMSNorm, Phi3RotaryEmbedding  # 必要に応じて修正
from .phi3 import Phi3MLP, Phi3Attention  # 必要に応じて修正


class LayerSelection(nn.Module):
    """
    A network that selects one layer or residual connection from a set of layers.
    Gumbel-Softmax trick を利用して、確率的かつ微分可能な選択を行う.
    """

    def __init__(self, num_layers: int, hidden_size: int, temperature: float = 1.0):
        super().__init__()
        self.selector = nn.Linear(
            hidden_size, num_layers + 1
        )  # +1 for residual connection
        self.num_layers = num_layers
        self.temperature = temperature  # Gumbel-Softmax temperature

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: The output from the previous layer.

        Returns:
            A tensor of shape (batch_size, num_layers + 1) representing the selection probabilities,
            and a one-hot tensor of shape (batch_size, num_layers + 1) representing the selected layer.
        """
        logits = self.selector(hidden_states)
        # Gumbel-Softmax
        selection_probs = F.gumbel_softmax(
            logits, tau=self.temperature, hard=True, dim=-1
        )
        # hard=True: one-hotベクトルを返す。推論時はこちらを利用
        # # hard=False: one-hotベクトルに近似された連続的なベクトルを返す。学習時はこちらを利用
        return selection_probs


class SelectiveDecoderLayer(nn.Module):
    def __init__(self, config: Phi3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Phi3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Phi3MLP(config)
        self.input_layernorm = Phi3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Phi3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.config = config
        self.resid_attn_dropout = nn.Dropout(config.resid_pdrop)
        self.resid_mlp_dropout = nn.Dropout(config.resid_pdrop)
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        attn_output = attn_outputs[0]
        self_attn_weights = attn_outputs[1] if output_attentions else None

        attn_output = self.resid_attn_dropout(attn_output)
        hidden_states = residual + attn_output  # residual connection.

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states)
        mlp_output = self.resid_mlp_dropout(mlp_output)
        hidden_states = residual + mlp_output

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class SelectiveModel(Phi3PreTrainedModel):
    def __init__(self, config: Phi3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                SelectiveDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Phi3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Phi3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Layer selection networks
        self.layer_selectors = nn.ModuleList(
            [
                LayerSelection(config.num_hidden_layers, config.hidden_size)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.num_hidden_layers = config.num_hidden_layers
        self.temperature = (
            config.gumbel_softmax_temperature
            if hasattr(config, "gumbel_softmax_temperature")
            else 1.0
        )

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_items_in_batch=None,
        **flash_attn_kwargs: FlashAttentionKwargs,  # Unpack[FlashAttentionKwargs] causes error in runtime
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            if past_key_values is not None and use_cache:
                # キャッシュ使用時、かつ position_ids が明示的に与えられていない場合:
                # cache_position を使う (past_key_values の最後の位置 + 1 から始まる)
                position_ids = cache_position.unsqueeze(0)
            else:
                # キャッシュを使用しない、または最初のステップの場合:
                # input_ids/inputs_embeds の長さに応じて 0 から始まる連番を作成
                position_ids = torch.arange(
                    0,
                    inputs_embeds.shape[1],
                    dtype=torch.long,
                    device=inputs_embeds.device,
                ).unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values,
            output_attentions,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        # Initialize past_key_values for each layer if not provided
        if past_key_values is None or len(past_key_values) == 0:
            past_key_values = [None] * len(self.layers)

        for i in range(self.num_hidden_layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # Layer Selection for the *next* layer
            selected_layer_one_hot = self.layer_selectors[i](hidden_states)

            # Residual connection handling
            residual = hidden_states

            batch_size = hidden_states.size(0)
            next_hidden_states = torch.zeros_like(hidden_states)

            for batch_idx in range(batch_size):
                current_selected_layer_idx = torch.argmax(
                    selected_layer_one_hot[batch_idx]
                ).item()
                print("batch_idx", batch_idx)
                print("current_selected_layer_idx", current_selected_layer_idx)
                if (
                    current_selected_layer_idx == self.num_hidden_layers
                ):  # Residual connection
                    next_hidden_states[batch_idx] = residual[batch_idx]
                else:
                    selected_layer = self.layers[current_selected_layer_idx]
                    current_position_ids = position_ids[
                        :,
                        cache_position[batch_idx] : cache_position[batch_idx]
                        + hidden_states.size(1),
                    ]

                    if self.gradient_checkpointing and self.training:
                        layer_outputs = self._gradient_checkpointing_func(
                            selected_layer.__call__,
                            hidden_states[batch_idx].unsqueeze(
                                0
                            ),  # Process one batch item
                            (
                                causal_mask[batch_idx].unsqueeze(0)
                                if causal_mask is not None
                                else None
                            ),
                            current_position_ids,  # position_ids[batch_idx].unsqueeze(0),
                            past_key_values[current_selected_layer_idx],
                            output_attentions,
                            use_cache,
                            cache_position,
                            position_embeddings,
                        )
                    else:
                        layer_outputs = selected_layer(
                            hidden_states=hidden_states[batch_idx].unsqueeze(
                                0
                            ),  # Process one batch item
                            attention_mask=(
                                causal_mask[batch_idx].unsqueeze(0)
                                if causal_mask is not None
                                else None
                            ),
                            position_ids=current_position_ids,  # position_ids[batch_idx].unsqueeze(0),
                            past_key_value=past_key_values[current_selected_layer_idx],
                            output_attentions=output_attentions,
                            use_cache=use_cache,
                            cache_position=cache_position,
                            position_embeddings=position_embeddings,
                            **flash_attn_kwargs,
                        )

                    next_hidden_states[batch_idx] = layer_outputs[0]

                    if output_attentions:
                        if all_self_attns is None or all_self_attns == ():
                            all_self_attns = [None] * self.num_hidden_layers
                        if all_self_attns[current_selected_layer_idx] is None:
                            all_self_attns[current_selected_layer_idx] = []
                        all_self_attns[current_selected_layer_idx].append(
                            layer_outputs[1]
                        )

            hidden_states = next_hidden_states

        hidden_states = self.norm(hidden_states)
        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # stack attention
        if output_attentions:
            all_self_attns = [
                torch.cat(layer_attns, dim=0) if layer_attns is not None else None
                for layer_attns in all_self_attns
            ]

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

        return output if return_dict else output.to_tuple()

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and past_key_values is not None:
                is_padding_right = (
                    attention_mask[:, -1].sum().item() != input_tensor.size()[0]
                )
                if is_padding_right:
                    raise ValueError(
                        "You are attempting to perform batched generation with padding_side='right'"
                        " this may lead to unexpected behaviour for Flash Attention version of Phi3. Make sure to "
                        " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                    )
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        using_static_cache = isinstance(past_key_values, StaticCache)
        using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if (
            self.config._attn_implementation == "sdpa"
            and not (using_static_cache or using_sliding_window_cache)
            and not output_attentions
        ):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                sliding_window=self.config.sliding_window,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        # SlidingWindowCache or StaticCache
        if using_sliding_window_cache or using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        # DynamicCache or no cache
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            config=self.config,
            past_key_values=past_key_values,
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(
                causal_mask, min_dtype
            )

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        config: Phi3Config,
        past_key_values: Cache,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to plcae the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
            config (`Phi3Config`):
                The model's configuration class
            past_key_values (`Cache`):
                The cache class that is being used currently to generate
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length),
                fill_value=min_dtype,
                dtype=dtype,
                device=device,
            )
            diagonal_attend_mask = torch.arange(
                target_length, device=device
            ) > cache_position.reshape(-1, 1)
            if config.sliding_window is not None:
                # if we have sliding window, we should not attend to tokens beyond sliding window length, so we mask them out also
                # the check is needed to verify is current checkpoint was trained with sliding window or not
                if (
                    not isinstance(past_key_values, SlidingWindowCache)
                    or sequence_length > target_length
                ):
                    sliding_attend_mask = torch.arange(
                        target_length, device=device
                    ) <= (cache_position.reshape(-1, 1) - config.sliding_window)
                    diagonal_attend_mask.bitwise_or_(sliding_attend_mask)
            causal_mask *= diagonal_attend_mask
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = (
                    causal_mask.clone()
                )  # copy to contiguous memory for in-place edit
                if attention_mask.shape[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = attention_mask.shape[-1]
                padding_mask = (
                    causal_mask[:, :, :, :mask_length]
                    + attention_mask[:, None, None, :]
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[
                    :, :, :, :mask_length
                ].masked_fill(padding_mask, min_dtype)
        return causal_mask


class SelectiveForCausalLM(Phi3ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = SelectiveModel(config)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        num_logits_to_keep=None,
        **kwargs,
    ):
        # Overwritten -- this model may need to switch between short and long rope, invalidating the cache in the
        # process

        # When the first time input length reached long and short factor switching point, enforce re-compute cache
        # It will cause downside of slower at this single token position, however, better than current failure.
        if (
            past_key_values
            and self.config.rope_scaling
            and input_ids.shape[1] >= self.config.original_max_position_embeddings + 1
        ):
            past_length = cache_position[0]
            if past_length <= self.config.original_max_position_embeddings:
                past_key_values = None

        model_inputs = super().prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            num_logits_to_keep=num_logits_to_keep,
            **kwargs,
        )
        return model_inputs


if __name__ == "__main__":
    config = Phi3Config(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=3,  # Keep it small for testing
        num_attention_heads=3,
        intermediate_size=32,
        hidden_act="silu",
        max_position_embeddings=2048,
        rms_norm_eps=1e-05,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        _attn_implementation="eager",
    )

    model = SelectiveForCausalLM(config)

    # Create some dummy inputs
    batch_size = 2
    sequence_length = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, sequence_length))
    attention_mask = torch.ones((batch_size, sequence_length), dtype=torch.long)

    # Forward pass
    outputs = model(
        input_ids=input_ids, attention_mask=attention_mask, output_attentions=True
    )
    print(outputs)
