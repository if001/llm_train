import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer


# from analyzer.context_mapper.model import PrefixMapper
from model import PrefixMapper

class Inference():
    def __init__(self):
        # encoder_model_name = "bert-base-uncased"
        # self.context_tokenizer = AutoTokenizer.from_pretrained(encoder_model_name)
        # self.context_encoder = AutoModel.from_pretrained(encoder_model_name)
        self.context_encoder = SentenceTransformer("cl-nagoya/ruri-base-v2")


        decoder_model_name = "microsoft/Phi-4-mini-instruct"
        # decoder_model_name = "microsoft/phi-4"
        self.decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_model_name)
        self.decoder_model = AutoModelForCausalLM.from_pretrained(decoder_model_name, trust_remote_code=True)
        self.decoder_model.eval()  # 重みは固定

    def inference(self, context, question):
        # BERTの出力 → GPT2のembedding空間に変換
        # mapper = PrefixMapper(768, self.decoder_model.config.n_embd, prefix_len=1)
        mapper = PrefixMapper.load("./prefix_mapper.pt")

        # ---- Step 3: 入力テキスト ----
        text_A = context
        text_B = question

        # ---- Step 4: 文章A → BERTでベクトル化 ----
        with torch.no_grad():
            # inputs_A = self.context_tokenizer(text_A, return_tensors="pt")
            output_A = self.context_encoder(text_A).last_hidden_state  # (1, seq_len, 768)
            context_embedding = output_A.mean(dim=1)  # 平均プーリングで1ベクトルに (1, 768)

        # ---- Step 5: ベクトルをembedding空間へマッピング ----
        mapped_context = mapper(context_embedding)  # (1, hidden_dim)

        # ---- Step 6: 文章B → トークン化し、embedding取得 ----
        inputs_B = self.decoder_tokenizer(text_B, return_tensors="pt")
        input_ids = inputs_B["input_ids"]
        with torch.no_grad():
            B_embeddings = self.decoder_model.transformer.wte(input_ids)

        # ---- Step 7: ベクトルをprefixとして前に追加 ----
        # mapped_contextは (1, hidden_dim) → (1, 1, hidden_dim)
        prefix = mapped_context.unsqueeze(1)
        combined_embeddings = torch.cat([prefix, B_embeddings], dim=1)

        # ---- Step 8: デコーダにembeddingで入力（forward時にinputs_embedsを使用） ----
        outputs = self.decoder_model(inputs_embeds=combined_embeddings)
        logits = outputs.logits

        # ---- Step 9: 出力トークンを生成 ----
        predicted_ids = torch.argmax(logits, dim=-1)
        print(self.decoder_tokenizer.batch_decode(predicted_ids, skip_special_tokens=True))


if __name__ == "__main__":
    context = "The capital of France is Paris."
    question = "What is the capital of France?"
    
    inference = Inference()
    inference.inference(context, question)