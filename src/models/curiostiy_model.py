from typing import Callable, List, Optional, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaModel, LlamaTokenizer, LlamaConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import (
    Cache,
    DynamicCache,
    SlidingWindowCache,
    StaticCache,
)

from .phi3 import (
    Phi3ForCausalLM,
    Phi3MLP,
    Phi3Attention,
    Phi3RMSNorm,
    Phi3RotaryEmbedding,
    Phi3Config,
    Phi3Model,
    Phi3PreTrainedModel,
    KwargsForCausalLM,
)

class PrimaryLPM(nn.Module):
    def __init__(self, hidden_dim, embedding_dim=None, use_prev_token=False):
        super().__init__()
        self.use_prev_token = use_prev_token
        input_dim = hidden_dim
        if use_prev_token:
            input_dim += embedding_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, h_t, y_prev=None):
        """
        h_t: 時刻tにおけるbaseのLLMの最終attention層の出力
        y_prev: 時刻tにおける正解ラベルの埋め込みベクトル
        """

        if self.use_prev_token:
            if y_prev is None:
                raise ValueError("use_prev_token=True requires y_prev input.")
            h_t = torch.cat([h_t, y_prev], dim=-1)
        l_hat_t = self.mlp(h_t).squeeze(-1) # [batch_size, seq_len]
        return l_hat_t

class SecondaryLPM(nn.Module):
    def __init__(self, hidden_dim, embedding_dim=None, use_next_token=False):
        super().__init__()
        self.use_next_token = use_next_token
        input_dim = hidden_dim + 1 # l_hat_tの次元を1とする
        if use_next_token:
            input_dim += embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, l_hat_t, h_next, y_next=None):
        l_hat_t = l_hat_t.unsqueeze(-1)  # MLPの入力次元に合わせる
        if self.use_next_token:
            if y_next is None:
                raise ValueError("use_next_token=True requires y_next input.")
            h_next = torch.cat([h_next, y_next],dim=-1)
        delta_l_hat_t = self.mlp(torch.cat([l_hat_t, h_next], dim=-1)).squeeze(-1) # [batch, seq_len]
        return delta_l_hat_t


class CuriosityModel(Phi3Model):
    def __init__(self, config: LlamaConfig, k=1, use_prev_token_lpm=False, use_next_token_lpm=False):
        super().__init__(config)

        self.k = k
        self.primary_lpm = PrimaryLPM(config.hidden_size, config.hidden_size, use_prev_token=use_prev_token_lpm)
        self.secondary_lpm = SecondaryLPM(config.hidden_size, config.hidden_size, use_next_token=use_next_token_lpm)
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.pad_token_id = config.pad_token_id

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,  # 隠れ状態を必ず出力
            return_dict=return_dict,
        )

        h_t = outputs.hidden_states[-1]  # 最終層の隠れ状態

        if labels is not None: # labelsがあるときだけy_prev, y_nextを計算
            replaced_labels = torch.where(labels == -100, self.pad_token_id, labels)
            y_prev = self.embed_tokens(replaced_labels)
            y_next = self.embed_tokens(torch.roll(replaced_labels, shifts=-self.k, dims=1))

            # padding部分ではy_nextを計算しない
            padding_mask = (replaced_labels == self.padding_idx)
            rolled_padding_mask = torch.roll(padding_mask, shifts=-self.k, dims=1)
            y_next = torch.where(rolled_padding_mask.unsqueeze(-1), torch.zeros_like(y_next), y_next)
        else:
            y_prev = None
            y_next = None

        l_hat_t = self.primary_lpm(h_t, y_prev)


        # Secondary LPM
        if attention_mask is not None:
            shifted_attention_mask = torch.roll(attention_mask, shifts=-self.k, dims=1)
            h_next = torch.where(shifted_attention_mask.unsqueeze(-1).bool(), outputs.hidden_states[-1], torch.zeros_like(outputs.hidden_states[-1]))
        else:
            h_next = outputs.hidden_states[-1]

        delta_l_hat_t = self.secondary_lpm(l_hat_t, h_next, y_next)

        return outputs, l_hat_t, delta_l_hat_t

@dataclass
class CuriosityModelCausalLMOutputWithPast(CausalLMOutputWithPast):
    primary_loss = None
    secondary_loss = None
    l_hat_t = None
    delta_l_hat_t = None

class CuriosityModelForCausalLM(Phi3ForCausalLM):
    def __init__(self, config, k=1, use_prev_token_lpm=False, use_next_token_lpm=False):
        super().__init__(config)
        self.model = CuriosityModel(config, k=k, use_prev_token_lpm=use_prev_token_lpm, use_next_token_lpm=use_next_token_lpm)
        self.loss_fn = nn.CrossEntropyLoss(reduction='none') #loss算出に用いるのでconfigから独立させる

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs, l_hat_t, delta_l_hat_t = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.lm_head(outputs.last_hidden_state)

        lm_loss = None
        primary_loss = None
        secondary_loss = None

        if labels is not None:
            # base model loss
            lm_loss = self.loss_fn(logits.view(-1, self.config.vocab_size), labels.view(-1))
            lm_loss = lm_loss.view(logits.size(0), logits.size(1))

            # Primary LPM loss
            primary_loss = torch.mean((l_hat_t - lm_loss) ** 2)

            # Secondary LPM loss
            l_t_plus_k = torch.roll(lm_loss, shifts=-self.model.k, dims=1)
             # padding部分ではlossを計算しない
            if attention_mask is not None:
                shifted_attention_mask = torch.roll(attention_mask, shifts=-self.model.k, dims=1)
                l_t_plus_k = torch.where(shifted_attention_mask.bool(), l_t_plus_k, torch.zeros_like(l_t_plus_k))

            secondary_loss = torch.mean((delta_l_hat_t - (l_t_plus_k - l_hat_t)) ** 2)


        if not return_dict:
            output = (logits, outputs.past_key_values, outputs.hidden_states, outputs.attentions, l_hat_t, delta_l_hat_t)
            return ((lm_loss, primary_loss, secondary_loss) + output) if lm_loss is not None else output

        return CuriosityModelCausalLMOutputWithPast(
            loss=lm_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            primary_loss=primary_loss,
            secondary_loss=secondary_loss,
            l_hat_t=l_hat_t,
            delta_l_hat_t=delta_l_hat_t,
        )
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs
        )
        return model_inputs