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


class QADataCollator:
    """
    * input_ids, q_len, sentence_vec がある Dict の list を受け取り
    * - tokenizer.pad で input_ids / attention_mask をバッチ化
    * - labels を生成し、質問部分 (q_len) を ignore_index(-100) でマスク
    * - sentence_vec (B, D) をバッチに戻す
    """

    def __init__(
        self,
        tokenizer,
        ignore_index: int = -100,
        padding: str = "longest",
    ):
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
        self.padding = padding

    def __call__(self, features):
        # 1) sentence_vec と q_len を抜き出し
        sentence_vecs = []
        q_lens = []
        for f in features:
            vec = f.pop("sentence_vec")
            # HF datasets では list で渡ってくるので tensor に変換
            if not torch.is_tensor(vec):
                vec = torch.tensor(vec, dtype=torch.float)
            sentence_vecs.append(vec)
            q_lens.append(f.pop("q_len"))

        sentence_vecs = torch.stack(sentence_vecs)

        # 2) tokenizer.pad で input_ids / attention_mask をテンソル化
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            return_tensors="pt",
        )  # -> {"input_ids", "attention_mask", ...}

        # 3) labels = input_ids.clone() し、質問部分を -100 でマスク
        labels = batch["input_ids"].clone()
        for i, q_len in enumerate(q_lens):
            labels[i, :q_len] = self.ignore_index

        # 4) 追加フィールドをバッチに戻す
        batch["labels"] = labels
        batch["sentence_vec"] = sentence_vecs
        return batch


# 1. components ---------------------------------------------------------------
# st_model_name = "sentence-transformers/all-MiniLM-L6-v2"
st_model_name = "cl-nagoya/ruri-base-v2"

# lm_name = "microsoft/Phi-4-mini-instruct"
lm_name = "google/gemma-3-1b-pt"

sentence_encoder = SentenceTransformer(st_model_name)
tokenizer = AutoTokenizer.from_pretrained(lm_name)
processor = STContextProcessor(sentence_encoder, tokenizer, max_length=1024)

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
    return processor(context, query, answer)


ds = raw_ds.map(preprocess, remove_columns=raw_ds.column_names)
ds = ds.train_test_split(test_size=0.1)
# 3. train --------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir="st_blip2_ckpt",
    per_device_train_batch_size=16,
    learning_rate=5e-4,
    num_train_epochs=1,
    bf16=True,
    save_total_limit=2,
    report_to="wandb",
    remove_unused_columns=False,
    logging_strategy="steps",
    logging_steps=50,
)

data_collator = QADataCollator(tokenizer)
# data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    data_collator=data_collator,
)

trainer.train()
