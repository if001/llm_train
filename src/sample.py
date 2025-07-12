import torch
from models.phi3_config import Phi3Config
from models.selective_phi3_v2 import SelectiveForCausalLM


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
    # print(outputs)
