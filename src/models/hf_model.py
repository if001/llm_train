from transformers import Phi3ForCausalLM, Phi3Config, Qwen2ForCausalLM, Qwen2Config
from models.selective_phi3_v2 import SelectiveForCausalLM
from models.curiostiy_model import CuriosityModelForCausalLM


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
        return CuriosityModelForCausalLM(Phi3Config(**config))
    else:
        raise ValueError("not impl hf models: ", model_name)
