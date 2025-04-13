import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset

from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

def get_qwen():
    model_name = "Qwen/Qwen2-0.5B"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    # tokenizer = AutoTokenizer.from_pretrained(model_name)

    # configと既存のlayer取得
    config = model.config
    # original_layers = model.transformer.layers  # nn.ModuleList of Qwen2DecoderLayer (24層)

    # 25番目の層を新規にランダム初期化で作成（layer_idx=24）
    new_layer = Qwen2DecoderLayer(config, layer_idx=24)

    # 25番目として追加
    model.model.layers.append(new_layer)
    for i, layer in enumerate(model.model.transformer.layers):
        for param in layer.parameters():
            param.requires_grad = (i == 24)

    model.config.max_window_layers = 25
    model.config.num_hidden_layers = 25
    return model
