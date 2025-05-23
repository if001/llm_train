import torch
from transformers import AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint

from model import ContextBLIP2Wrapper
from context_processor import STContextProcessor
from sentence_transformers import SentenceTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"

# lm_name = "microsoft/Phi-4-mini-instruct"
# lm_name = "google/gemma-3-1b-pt"
lm_name = "google/gemma-3-1b-it"
st_name = "cl-nagoya/ruri-base-v2"

ckpt_dir = get_last_checkpoint("st_blip2_ckpt")
model = ContextBLIP2Wrapper.from_pretrained(ckpt_dir).to(device)
tokenizer = AutoTokenizer.from_pretrained(lm_name)
processor = STContextProcessor(SentenceTransformer(st_name), tokenizer)

# ----- user query ------------------------------------------------------------
context_txt = "量子コンピュータの誕生と現在の課題をまとめた論文。"
prompt_txt = "将来の量子AIについて一段落で説明してください。"

context_txt="ジョンは犬を飼っているが、猫は飼っていない。"
prompt_txt="ジョンはどんな動物を飼っていますか？"

batch = processor(context_txt, prompt_txt, "")
batch = {k: v.unsqueeze(0).to(device) for k, v in batch.items()}

generated_ids = model.generate_with_context(
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
    sentence_vec=batch["sentence_vec"],
    max_new_tokens=50,
)

result = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print("--- gen ---")
print(result)