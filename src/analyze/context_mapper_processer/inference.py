import torch
from transformers import AutoTokenizer

from model import ContextBLIP2Wrapper
from context_processor import STContextProcessor
from sentence_transformers import SentenceTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"

lm_name = "microsoft/Phi-4-mini-instruct"
st_name = "cl-nagoya/ruri-base-v2"

model = ContextBLIP2Wrapper.from_pretrained("st_blip2_ckpt").to(device)
tokenizer = AutoTokenizer.from_pretrained(lm_name)
processor = STContextProcessor(SentenceTransformer(st_name), tokenizer)

# ----- user query ------------------------------------------------------------
context_txt = "量子コンピュータの誕生と現在の課題をまとめた論文。"
prompt_txt = "将来の量子AIについて一段落で説明してください。"

batch = processor(context_txt, prompt_txt)
batch = {k: v.to(device) for k, v in batch.items()}

generated_ids = model.generate_with_context(
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
    sentence_vec=batch["sentence_vec"],
    max_new_tokens=80,
)

print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
