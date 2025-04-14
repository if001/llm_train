from transformers import Phi3ForCausalLM, Phi3Config, Qwen2ForCausalLM, Qwen2Config
from models.selective_phi3_v2 import SelectiveForCausalLM
from models.curiostiy_model import (
    CuriosityModelForCausalLM,
    CuriosityModelConfig
)
from models.few_attention_model import (
    FewAttentionModelForCausalLM,
    FewAttentionConfig
)
from models.qwen2_fixed_layer import get_qwen

class Phi3(Phi3ForCausalLM):
    def __init__(self, config):
        super().__init__(Phi3Config(**config))


def get_hf_models(config):
    if "name" not in config:
        raise ValueError("config must have name field")
    model_name = config["name"]
    if "phi3" in model_name:
        return Phi3(config)
    if "selective_v2" in model_name:
        return SelectiveForCausalLM(Phi3Config(**config))
    if "curiosity" in model_name:
        return CuriosityModelForCausalLM(CuriosityModelConfig(**config))
    if "few_attention" in model_name:
        return FewAttentionModelForCausalLM(FewAttentionConfig(**config))
    if "qwen2_0.5b_25" in model_name:
        return get_qwen(model_name)
    if "qwen2_0.5b_24" in model_name:
        return get_qwen(model_name)
    if "qwen2_0.5b_24_moe" in model_name:
        return get_qwen(model_name)
    else:
        raise ValueError("not impl hf models: ", model_name)
