from transformers import Phi3ForCausalLM, Phi3Config
# from transformers import Qwen2ForCausalLM, Qwen2Config

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
        from models.selective_phi3_v2 import SelectiveForCausalLM
        return SelectiveForCausalLM(Phi3Config(**config))
    if "curiosity" in model_name:
        from models.curiostiy_model import (
            CuriosityModelForCausalLM,
            CuriosityModelConfig
        )
        return CuriosityModelForCausalLM(CuriosityModelConfig(**config))
    if "few_attention" in model_name:
        from models.few_attention_model import (
            FewAttentionModelForCausalLM,
            FewAttentionConfig
        )
        return FewAttentionModelForCausalLM(FewAttentionConfig(**config))
    if "qwen2_0.5b_25" in model_name:
        return get_qwen(model_name)
    if "qwen2_0.5b_24" in model_name:
        return get_qwen(model_name)
    if "qwen2_0.5b_24_moe" in model_name:
        return get_qwen(model_name)
    if "residual-tiny" in model_name or "residual-small" in model_name:
        from models.residual_diff import (
            ResidualNetConfig,
            ResidualNetForCausalLM,
        )        
        return ResidualNetForCausalLM(ResidualNetConfig(**config))
    if "residual-v2-tiny" in model_name:
        from models.residual_diff_v2 import (
            ResidualNetV2Config,
            ResidualNetV2ForCausalLM,
        )
        return ResidualNetV2ForCausalLM(ResidualNetV2Config(**config))
    if "residual-v3-tiny" in model_name:
        from models.residual_diff_v2 import (
            ResidualNetV2Config,
            ResidualNetV3ForCausalLM,
        )
        return ResidualNetV3ForCausalLM(ResidualNetV2Config(**config))
    if "conv-tiny" in model_name:
        from models.conv_attn import (
            PyramidPhi3Config,
            PyramidPhi3ForCausalLM
        )
        return PyramidPhi3ForCausalLM(PyramidPhi3Config(**config))
    else:
        raise ValueError("not impl hf models: ", model_name)

