import torch
from torch import nn
from typing import Optional, Tuple, List, Union

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_attn_mask_utils import AttentionMaskConverter

from transformers.utils import logging

# from transformers.models.phi3.modeling_phi3 import (
#     Phi3MLP,
#     Phi3Attention,
#     Phi3RMSNorm,
#     Phi3RotaryEmbedding,
#     apply_rotary_pos_emb,
#     eager_attention_forward,
# )

from models.phi3_config import Phi3Config
from models.phi3 import (
    Phi3PreTrainedModel,
    Phi3ForCausalLM,
    Phi3MLP,
    Phi3Attention,
    Phi3RMSNorm,
    Phi3RotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

logger = logging.get_logger(__name__)


class AttentionSelector(nn.Module):
    def __init__(self, config, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.selector = nn.Linear(
            config.hidden_size, num_layers + 1
        )  # +1 for residual connection

    def forward(self, hidden_states):
        attn_weights = self.selector(hidden_states)
        attn_probs = torch.softmax(attn_weights, dim=-1)
        return attn_probs


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

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
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
            **kwargs,
        )
        attn_output = attn_outputs[0]
        hidden_states = residual + self.resid_attn_dropout(attn_output)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.resid_mlp_dropout(hidden_states)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_outputs[1],)  # Add attention weights if needed

        return outputs


class SelectiveTransformer(nn.Module):
    def __init__(self, config: Phi3Config):
        super().__init__()
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

        self.attention_selectors = nn.ModuleList(
            [
                AttentionSelector(config, config.num_hidden_layers)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.config = config  #  config

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs,
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

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        if cache_position is None:
            past_seen_tokens = (
                0 if past_key_values[0] is None else past_key_values[0][0].shape[2]
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

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

        for i in range(self.config.num_hidden_layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            selector_probs = self.attention_selectors[i](hidden_states)

            selected_outputs = []
            for j, layer in enumerate(self.layers):
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values[j] if past_key_values else None,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )
                selected_outputs.append(layer_outputs[0])

            residual_output = hidden_states
            selected_outputs.append(residual_output)  # residual connection

            selected_outputs = torch.stack(
                selected_outputs, dim=-1
            )  # Stack along a new dimension

            hidden_states = torch.sum(
                selected_outputs * selector_probs.unsqueeze(-2), dim=-1
            )

            if output_attentions:
                # Need to modify here if you want correct attention weights
                all_self_attns += (None,)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=(
                past_key_values if use_cache else None
            ),  # You might need to modify this
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: List[torch.FloatTensor],  # Changed to List
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if (
                attention_mask is not None
                and past_key_values is not None
                and any(past_key_values)
            ):  # Modified
                is_padding_right = (
                    attention_mask[:, -1].sum().item() != input_tensor.size()[0]
                )
                if is_padding_right:
                    raise ValueError(
                        "You are attempting to perform batched generation with padding_side='right'"
                        " this may lead to unexpected behaviour for Flash Attention version of Phi3. Make sure to "
                        " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                    )
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        past_seen_tokens = (
            past_key_values[0][0].shape[2] if past_key_values[0] is not None else 0
        )  # Modified

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        target_length = past_seen_tokens + sequence_length + 1

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
            past_key_values=past_key_values,  # Modified
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu"]
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
        past_key_values: List[torch.FloatTensor],  # Changed to List
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
                The target length.
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
                past key values
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


class SelectivePhi3ForCausalLM(Phi3ForCausalLM):  # Inherit from your causal LM class
    def __init__(self, config):
        super(Phi3PreTrainedModel, self).__init__(
            config
        )  # Initialize as PreTrainedModel
        self.model = SelectiveTransformer(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

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
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # logits_to_keep: Union[int, torch.Tensor] = 0,
        num_logits_to_keep: int = 0,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).
        Returns:

        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        # slice_indices = (
        #     slice(-logits_to_keep, None)
        #     if isinstance(logits_to_keep, int)
        #     else logits_to_keep
        # )
        # logits = self.lm_head(hidden_states[:, slice_indices, :])
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1)
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,  # add
        use_cache=True,
        num_logits_to_keep=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if (
            past_key_values is not None and past_key_values[0] is not None
        ):  # Modified this line
            input_ids = input_ids[:, -1:]

        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        input_shape = input_ids.shape
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)
        if position_ids is None:
            if past_key_values is not None and past_key_values[0] is not None:
                past_length = past_key_values[0][0].shape[2]
                position_ids = input_ids.new_zeros(1, input_shape[-1]).fill_(
                    past_length
                )
            else:
                position_ids = input_ids.new_zeros(1, input_shape[-1])
                torch.arange(
                    0, input_shape[-1], dtype=torch.long, device=input_ids.device
                )

        # We additionally need cache_position, which is constructed as follows:
        # - If this is the first step (no cache), the cache position starts at 0
        # - If the cache is static, the cache position will increase along with input_ids.shape[-1]
        # - If the cache is dynamic (default), the cache position will be the length of the cache
        if cache_position is None:
            if (
                past_key_values is not None and past_key_values[0] is not None
            ):  # Modified this line
                cache_position = torch.tensor(
                    [kv[0].shape[2] for kv in past_key_values], device=input_ids.device
                )  # Modified this line
            else:
                cache_position = torch.tensor([0], device=input_ids.device)

        if inputs_embeds is not None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "cache_position": cache_position,
            }
        )
        return model_inputs
