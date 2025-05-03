from transformers import ProcessorMixin
from sentence_transformers import SentenceTransformer
from typing import List


class STContextProcessor(ProcessorMixin):
    attributes = ["sentence_encoder", "tokenizer"]

    def __init__(
        self, sentence_encoder: SentenceTransformer, tokenizer, max_length=256
    ):
        self.sentence_encoder = sentence_encoder
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(
        self, context: str | List[str], text: str | List[str], return_tensors="pt"
    ):
        """
        * context – string(s) fed to SentenceTransformer
        * text    – string(s) fed to the LLM tokenizer
        """
        ctx_vec = self.sentence_encoder.encode(
            context, convert_to_tensor=True, show_progress_bar=False
        )  # [B, Dvec] (or [Dvec] if single)
        tok = self.tokenizer(
            text,
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            return_tensors=return_tensors,
        )
        tok["sentence_vec"] = ctx_vec
        return tok

    # mandatory aliases
    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)
