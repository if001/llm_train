from typing import Optional, Tuple

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset

from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2RMSNorm, Qwen2MLP

class Qwen2MoEDecoderLayer(nn.Module):
    def __init__(self, config, pretrained_layer: nn.Module, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        # 事前学習済みのAttentionとExpert1をコピー（重み固定）
        self.self_attn = pretrained_layer.self_attn
        self.expert1 = pretrained_layer.mlp

        # ランダム初期化のExpert2とRouter
        self.expert2 = Qwen2MLP(config)
        self.router = nn.Linear(config.hidden_size, 2)

        self.input_layernorm = pretrained_layer.input_layernorm
        self.post_attention_layernorm = pretrained_layer.post_attention_layernorm

        # Expert1とAttentionは固定（requires_grad=False）
        for param in self.self_attn.parameters():
            param.requires_grad = False
        for param in self.expert1.parameters():
            param.requires_grad = False
        for param in self.input_layernorm.parameters():
            param.requires_grad = False
        for param in self.post_attention_layernorm.parameters():
            param.requires_grad = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Attention（事前学習済み）
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
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # Routing
        router_logits = self.router(hidden_states)
        router_probs = F.softmax(router_logits, dim=-1)
        expert_choice = torch.argmax(router_probs, dim=-1)

        expert1_mask = (expert_choice == 0).unsqueeze(-1).float()
        expert2_mask = (expert_choice == 1).unsqueeze(-1).float()

        expert1_out = self.expert1(hidden_states)
        expert2_out = self.expert2(hidden_states)

        moe_output = expert1_mask * expert1_out + expert2_mask * expert2_out
        hidden_states = residual + moe_output

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs

def get_qwen(model_name):
    # model_name = "Qwen/Qwen2-0.5B"
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B")

    config = model.config
    for param in model.model.embed_tokens.parameters():
        param.requires_grad = False

    for param in model.model.norm.parameters():
        param.requires_grad = False

    if "qwen2_0.5b_25" in model_name:
        new_layer = Qwen2DecoderLayer(config, layer_idx=24)
        model.model.layers.append(new_layer)
        for i, layer in enumerate(model.model.layers):
            for param in layer.parameters():
                param.requires_grad = (i == 24)

        model.config.max_window_layers = 25
        model.config.num_hidden_layers = 25

    if "qwen2_0.5b_24" in model_name:
        for i, layer in enumerate(model.model.layers):
            for param in layer.parameters():
                param.requires_grad = False
            if i == 23:
                for param in layer.mlp.parameters():
                    param.requires_grad = False

    if "qwen2_0.5b_24_moe" in model_name:
        pretrained_layer = model.transformer.layers[23]
        model.transformer.layers[23] = Qwen2MoEDecoderLayer(config, pretrained_layer, layer_idx=23)

        for i, layer in enumerate(model.model.layers):
            for param in layer.parameters():
                param.requires_grad = (i == 23)

    trainable = [(name, p.numel()) for name, p in model.named_parameters() if p.requires_grad]
    print("Trainable parameters:")
    for name, count in trainable:
        print(f"{name}: {count:,}")
    return model
