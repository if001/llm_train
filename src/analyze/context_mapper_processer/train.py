import torch, datasets
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from sentence_transformers import SentenceTransformer

from model import ContextBLIP2Wrapper
from context_processor import STContextProcessor

# 1. components ---------------------------------------------------------------
# st_model_name = "sentence-transformers/all-MiniLM-L6-v2"
st_model_name = "cl-nagoya/ruri-base-v2"
# lm_name       = "gpt2"
lm_name = "microsoft/Phi-4-mini-instruct"
sentence_encoder = SentenceTransformer(st_model_name)
tokenizer = AutoTokenizer.from_pretrained(lm_name)
processor = STContextProcessor(sentence_encoder, tokenizer)

model = ContextBLIP2Wrapper(
    lm_name=lm_name,
    st_dim=sentence_encoder.get_sentence_embedding_dimension(),
    num_prefix_tokens=16,
    hidden_dim=512,
)

# 2. dummy data ---------------------------------------------------------------
# expects a dataset with {"context": ..., "input": ..., "output": ...}
# raw_ds = datasets.load_dataset("json", data_files="train.jsonl")  # your own file
raw_ds = datasets.load_dataset("cl-nagoya/auto-wiki-qa", split="train")  # your own file


def preprocess(ex):
    context = ex["text"]
    query = ex["query"]
    answer = ex["query"]
    proc = processor(context, query)
    labels = tokenizer(answer, truncation=True, max_length=256)["input_ids"]
    proc["labels"] = labels
    return proc


ds = raw_ds.map(preprocess, remove_columns=raw_ds.column_names)

# 3. train --------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir="st_blip2_ckpt",
    per_device_train_batch_size=4,
    learning_rate=5e-4,
    num_train_epochs=3,
    bf16=True,
    save_total_limit=2,
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    data_collator=data_collator,
)

trainer.train()
