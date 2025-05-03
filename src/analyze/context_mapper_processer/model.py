import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoConfig, PreTrainedModel

class STPrefixMapper(nn.Module):
    """
    Map a single SentenceTransformer vector to N pseudo-token embeddings.
    """
    def __init__(self, st_dim: int, lm_emb_dim: int,
                 num_tokens: int = 16, hidden_dim: int | None = None):
        super().__init__()
        self.num_tokens = num_tokens
        if hidden_dim:
            self.mapper = nn.Sequential(
                nn.Linear(st_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, lm_emb_dim * num_tokens)
            )
        else:  # simplest: single affine map
            self.mapper = nn.Linear(st_dim, lm_emb_dim * num_tokens)

    def forward(self, st_vec: torch.Tensor) -> torch.Tensor:
        # st_vec: [B, Dvec]
        B = st_vec.size(0)
        x = self.mapper(st_vec)                      # [B, N*Demb]
        return x.view(B, self.num_tokens, -1)        # [B, N, Demb]


class ContextBLIP2Wrapper(PreTrainedModel):
    """
    Wrap any causal-LM so it can consume a SentenceTransformer vector
    as a learned prefix (BLIP-2 Q-Former style).
    """
    config_class = AutoConfig  # lets us use from_pretrained easily

    def __init__(self,
                 lm_name: str = "gpt2",
                 st_dim: int = 768,
                 num_prefix_tokens: int = 16,
                 hidden_dim: int | None = None):
        config = AutoConfig.from_pretrained(lm_name)
        super().__init__(config)

        # ① backbone LM
        self.lm = AutoModelForCausalLM.from_pretrained(lm_name)
        lm_emb_dim = self.lm.get_input_embeddings().embedding_dim

        # ② tiny mapper
        self.prefix_mapper = STPrefixMapper(
            st_dim, lm_emb_dim, num_prefix_tokens, hidden_dim
        )
        self.num_prefix_tokens = num_prefix_tokens

        # ③ (optional) freeze LM to train only mapper
        for p in self.lm.parameters():
            p.requires_grad = False

    # ===== forward =====
    def forward(
        self,
        input_ids: torch.Tensor,            # [B, L]
        attention_mask: torch.Tensor | None = None,
        sentence_vec: torch.Tensor | None = None,   # [B, Dvec]
        labels: torch.Tensor | None = None,
    ):
        if sentence_vec is None:
            raise ValueError("Provide sentence_vec tensor")

        # a) build prefix embeddings
        prefix_embeds = self.prefix_mapper(sentence_vec)   # [B, N, Demb]

        # b) normal token embeddings
        tok_embeds = self.lm.get_input_embeddings()(input_ids)  # [B, L, Demb]

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

        # e) run LM
        return self.lm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

    # ===== generation helper =====
    @torch.no_grad()
    def generate_with_context(
        self,
        input_ids: torch.Tensor,
        sentence_vec: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **gen_kwargs
    ):
        prefix_embeds = self.prefix_mapper(sentence_vec)
        tok_embeds = self.lm.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([prefix_embeds, tok_embeds], dim=1)

        if attention_mask is not None:
            prefix_mask = torch.ones(
                (attention_mask.size(0), self.num_prefix_tokens),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        return self.lm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **gen_kwargs,
        )