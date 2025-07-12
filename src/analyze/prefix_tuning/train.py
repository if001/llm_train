from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import get_peft_model, PrefixTuningConfig, TaskType
import torch

# モデルとトークナイザの準備（GPT-2）
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # GPT2はpad_tokenがないのでeosを使う

model = AutoModelForCausalLM.from_pretrained(model_name)

# PEFTのPrefixTuning設定
peft_config = PrefixTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    num_virtual_tokens=10,
    prefix_projection=True,
)

model = get_peft_model(model, peft_config)

# データセットの準備（簡易的にwikitextで実験）
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]")

# トークナイズ関数
def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
# tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

# データコラレータ
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 学習設定
training_args = TrainingArguments(
    output_dir="./results_prefix",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=100,
    save_total_limit=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 学習
trainer.train()

# 保存
model.save_pretrained("./prefix-tuned-gpt2")
tokenizer.save_pretrained("./prefix-tuned-gpt2")