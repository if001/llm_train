def get_config(model_name):
    if model_name not in name_to_config:
        raise ValueError("model not impl")
    conf_dict = name_to_config[model_name]
    return conf_dict


configs = []


phi3 = [
    dict(
        name="phi3",
        vocab_size=50257,
        hidden_size=3072,
        intermediate_size=8192,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attention_dropout=0.0,
        hidden_act="silu",
        max_position_embeddings=4096,
        original_max_position_embeddings=4096,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        bos_token_id=1,
        eos_token_id=7,
        pad_token_id=7,
        sliding_window=None,
    ),
    dict(
        name="phi3-small",
        vocab_size=50257,  ## llm-jp
        hidden_size=896,
        intermediate_size=2048,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_key_value_heads=None,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attention_dropout=0.0,
        hidden_act="silu",
        max_position_embeddings=2048,
        original_max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        bos_token_id=1,
        eos_token_id=7,
        pad_token_id=7,
        sliding_window=None,
    ),
    dict(
        name="phi3-tiny",
        # vocab_size=65535, ## NovelAI/nerdstash-tokenizer-v2
        # vocab_size=96867, ## llm-jp-13b-v2
        # vocab_size=96877,  ## llm-jp-13b-v2 + 品詞トークン
        vocab_size=99574,  ## llm-jp/llm-jp-3-1.8b
        hidden_size=64,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=None,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attention_dropout=0.0,
        hidden_act="silu",
        max_position_embeddings=1024,
        original_max_position_embeddings=1024,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=4,
        sliding_window=None,
    ),
    dict(
        name="phi3-tiny-half",
        # vocab_size=65535, ## NovelAI/nerdstash-tokenizer-v2
        # vocab_size=96867,  ## llm-jp-13b-v2
        vocab_size=96877,  ## llm-jp-13b-v2 + 品詞トークン
        hidden_size=512,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=None,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attention_dropout=0.0,
        hidden_act="silu",
        max_position_embeddings=1024,
        original_max_position_embeddings=1024,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        bos_token_id=1,
        eos_token_id=7,
        pad_token_id=7,
        sliding_window=None,
    ),
]
configs.extend(phi3)


selective = [
    dict(
        name="selective_v2-tiny",
        # vocab_size=65535, ## NovelAI/nerdstash-tokenizer-v2
        # vocab_size=96867, ## llm-jp-13b-v2
        # vocab_size=96877,  ## llm-jp-13b-v2 + 品詞トークン
        vocab_size=99574,  ## llm-jp/llm-jp-3-1.8b
        hidden_size=64,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=None,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attention_dropout=0.0,
        hidden_act="silu",
        max_position_embeddings=4096,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=4,
        sliding_window=None,
    ),
]
configs.extend(selective)

name_to_config = {config["name"]: config for config in configs}
