import torch
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

        # self.sot = self.tokenizer.encode("<start_of_turn>", add_special_tokens=False)[0]
        # self.bos = tokenizer.encode("<bos>", add_special_tokens=False)[0]
        

    def __call__(
        self,
        context: str | List[str],
        question: str | List[str],
        answer: str | List[str],
        return_tensors="pt",
        chat_format = False
    ):
        """
        * context – string(s) fed to SentenceTransformer
        * text    – string(s) fed to the LLM tokenizer
        """
        ctx_vec = self.sentence_encoder.encode(
            context, convert_to_tensor=True, show_progress_bar=False
        )  # [B, Dvec] (or [Dvec] if single)

        if chat_format:
            t = self._tokenize(q, a)
            input_ids = t["input_ids"]
            attention_mask = torch.tensor([1] * len(input_ids))
        else:
            q_enc = self.tokenizer(
                question,
                add_special_tokens=False,
            )
            a_enc = self.tokenizer(
                answer,
                add_special_tokens=False,
            )
            eos_id = self.tokenizer.eos_token_id
            input_ids = torch.tensor(q_enc["input_ids"] + [eos_id] + a_enc["input_ids"] + [eos_id])

            attention_mask = torch.tensor([1] * len(input_ids))
            # labels = (
            #     [-100] * (len(q_enc["input_ids"]) + 1)  # 質問 + EOS
            #     + a_enc["input_ids"]  # 回答
            #     + [eos_id]  # 末尾 EOS
            # )
            q_len = len(q_enc["input_ids"]) + 1
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            # "labels": labels,
            # "q_len": q_len,
            "sentence_vec": ctx_vec,
        }

    def _tokenize(self, q, a):
        messages = [
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant.\n"},]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": question},]
                },
                {
                    "role": "system",
                    "content": [{"type": "text", "text": a },]
                },
            ],
        ]
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

    # mandatory aliases
    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)
