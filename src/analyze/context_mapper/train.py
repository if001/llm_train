import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import json
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AdamW
from sentence_transformers import SentenceTransformer

from model import PrefixMapper


# ---- 1. Dataset定義 ----
class QAWithContextDataset(Dataset):
    def __init__(self, path, question_tokenizer):
        with open(path, "r") as f:
            self.data = json.load(f)
        self.question_tokenizer = question_tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["question"]
        answer = item["answer"]

        full_text = question + " " + answer
        text_inputs = self.question_tokenizer(
            full_text, return_tensors="pt", truncation=True, padding=True
        )

        return {
            "input_ids": text_inputs["input_ids"].squeeze(0),
            "attention_mask": text_inputs["attention_mask"].squeeze(0),
        }


# ---- 3. 学習ループ ----
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tokenizer/Model読み込み
    # context_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # context_encoder = AutoModel.from_pretrained("bert-base-uncased").to(device)
    context_encoder = SentenceTransformer("cl-nagoya/ruri-base-v2")

    decoder_model_name = "microsoft/Phi-4-mini-instruct"
    # decoder_model_name = "microsoft/phi-4"
    decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_model_name)
    decoder_model = AutoModelForCausalLM.from_pretrained(
        decoder_model_name, trust_remote_code=True
    ).to(device)
    decoder_model.eval()  # GPT2は凍結

    # Dataset/DataLoader
    dataset = QAWithContextDataset("train_dataset.json", decoder_tokenizer)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Mapperの初期化
    mapper = PrefixMapper(768, decoder_model.config.n_embd, prefix_len=1).to(device)
    optimizer = AdamW(mapper.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(3):
        for batch in dataloader:
            # Step 1: Context → ベクトル化
            with torch.no_grad():
                ctx_input = {
                    k: v.squeeze(0).to(device)
                    for k, v in batch["context_inputs"].items()
                }
                context_vec = context_encoder(**ctx_input).last_hidden_state.mean(
                    dim=1
                )  # (B, 768)

            # Step 2: Mapper → Prefixベクトル
            prefix_embed = mapper(context_vec)  # (B, prefix_len, dim)

            # Step 3: 質問+答え → トークン化しembedding取得
            input_ids = batch["input_ids"].to(device)
            labels = input_ids.clone()

            with torch.no_grad():
                input_embed = decoder_model.transformer.wte(
                    input_ids
                )  # (B, seq_len, dim)

            # Step 4: Prefix追加
            full_embed = torch.cat([prefix_embed, input_embed], dim=1)

            # Step 5: モデルにembeddingを与えてforward
            outputs = decoder_model(inputs_embeds=full_embed)
            logits = outputs.logits[
                :, mapper.prefix_len :, :
            ]  # prefix部分を除外してLoss計算
            shift_logits = logits.contiguous().view(-1, logits.size(-1))
            shift_labels = labels.contiguous().view(-1)

            loss = loss_fn(shift_logits, shift_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")

    # 保存（任意）
    torch.save(mapper.state_dict(), "prefix_mapper.pt")


if __name__ == "__main__":
    train()
