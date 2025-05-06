import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoConfig, PreTrainedModel, PretrainedConfig


class STPrefixMapper(nn.Module):
    """
    Map a single SentenceTransformer vector to N pseudo-token embeddings.
    """

    def __init__(
        self,
        st_dim: int,
        lm_emb_dim: int,
        num_tokens: int = 16,
        hidden_dim: int | None = None,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        if hidden_dim:
            self.mapper = nn.Sequential(
                nn.Linear(st_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, lm_emb_dim * num_tokens),
            )
        else:  # simplest: single affine map
            self.mapper = nn.Linear(st_dim, lm_emb_dim * num_tokens)

    def forward(self, st_vec: torch.Tensor) -> torch.Tensor:
        # st_vec: [B, Dvec]
        B = st_vec.size(0)
        x = self.mapper(st_vec)  # [B, N*Demb]
        return x.view(B, self.num_tokens, -1)  # [B, N, Demb]

class ContextBlip2Config(PretrainedConfig):
    model_type = "context_blip2"

    def __init__(
        self,
        lm_name: str          = "gpt2",
        st_dim:  int          = 768,
        num_prefix_tokens: int = 16,
        hidden_dim: int | None = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.lm_name          = lm_name
        self.st_dim           = st_dim
        self.num_prefix_tokens = num_prefix_tokens
        self.hidden_dim       = hidden_dim


class ContextBLIP2Wrapper(PreTrainedModel):
    """
    Wrap any causal-LM so it can consume a SentenceTransformer vector
    as a learned prefix (BLIP-2 Q-Former style).
    """
    config_class = ContextBlip2Config

    def __init__(
        self,
        config: ContextBlip2Config
    ):

        super().__init__(config)

        # ① backbone LM
        # self.base_lm = AutoModelForCausalLM.from_pretrained(config.lm_name)
        # gemma3を使う場合
        self.base_lm = AutoModelForCausalLM.from_pretrained(config.lm_name, attn_implementation='eager')
        if hasattr(self.base_lm, "tie_weights"):
            self.base_lm.tie_weights()

        lm_emb_dim = self.base_lm.get_input_embeddings().embedding_dim

        # ② tiny mapper
        self.prefix_mapper = STPrefixMapper(
            config.st_dim,
            lm_emb_dim,
            config.num_prefix_tokens,
            config.hidden_dim,
        )

        self.num_prefix_tokens = config.num_prefix_tokens

        # ③ (optional) freeze LM to train only mapper
        for p in self.base_lm.parameters():
            p.requires_grad = False
        self.tie_weights()

    # ===== forward =====
    def forward(
        self,
        input_ids: torch.Tensor,  # [B, L]
        attention_mask: torch.Tensor | None = None,
        sentence_vec: torch.Tensor | None = None,  # [B, Dvec]
        labels: torch.Tensor | None = None,
    ):
        if sentence_vec is None:
            raise ValueError("Provide sentence_vec tensor")

        # a) build prefix embeddings
        prefix_embeds = self.prefix_mapper(sentence_vec)  # [B, N, Demb]

        # b) normal token embeddings
        tok_embeds = self.base_lm.get_input_embeddings()(input_ids)  # [B, L, Demb]

        # c) concat
        inputs_embeds = torch.cat([prefix_embeds, tok_embeds], dim=1)

        # d) extend mask
        if attention_mask is not None:
            prefix_mask = torch.ones(
                (attention_mask.size(0), self.num_prefix_tokens),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        B, L = input_ids.shape
        device = input_ids.device
        N = self.num_prefix_tokens
        if labels is not None:
            prefix_ignore = torch.full(
                (B, N), fill_value=-100, dtype=labels.dtype, device=device
            )
            labels = torch.cat([prefix_ignore, labels], dim=1)

        prefix_pos = torch.arange(0, N, device=device).unsqueeze(0).expand(B, -1)
        token_pos  = torch.arange(N, N + L, device=device).unsqueeze(0).expand(B, -1)
        position_ids = torch.cat([prefix_pos, token_pos], dim=1)     # [B, N+L]

        # e) run LM
        return self.base_lm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
        )

    # ===== generation helper =====
    @torch.no_grad()
    def generate_with_context(
        self,
        input_ids: torch.Tensor,
        sentence_vec: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **gen_kwargs,
    ):
        prefix_embeds = self.prefix_mapper(sentence_vec)
        tok_embeds = self.base_lm.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([prefix_embeds, tok_embeds], dim=1)

        if attention_mask is not None:
            prefix_mask = torch.ones(
                (attention_mask.size(0), self.num_prefix_tokens),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        return self.base_lm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **gen_kwargs,
        )

    def _filter_state_dict_for_save(self, state_dict):
        """lm. で始まるキーを落とす"""
        return {k: v for k, v in state_dict.items() if not k.startswith("base_lm.")}

    def save_pretrained(self, save_directory, **kwargs):
        super().save_pretrained(
            save_directory,
            self._filter_state_dict_for_save(self.state_dict()),
            **kwargs,
        )

    def tie_weights(self):
        """
        デフォルトのtie_weightがうまく動かないので無理やりセットする
        """
        self.base_lm.lm_head.weight = nn.Parameter(self.base_lm.model.embed_tokens.weight.clone())
