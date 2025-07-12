"""
文脈をkv_cacheとして渡し質問文に続く文章を生成する
文脈+質問文の結合したtext、質問文のみのtextでの生成と比較する

kv_cacheとして渡した場合と、結合して渡した場合でも結果は同じになるはず
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class Questions():
    def __init__(self, with_context = True, with_text = False):
        self.with_context = with_context
        self.with_text = with_text
        self._i = 0
        self.questions = [
            ["ジョンは犬を飼っているが、猫は飼っていない。","ジョンはどんな動物を飼っていますか？"],
            ["昨日は大雨だったが、今日は晴れている。", "今日の天気はどうですか？"],
            ["アリスはフランスに留学していたので、フランス語が堪能だ。","アリスは何語が得意ですか？"],
            ["本日の会議はリモートではなく、オフィスで行われる。","会議はどこで開かれますか？"],
            ["昨夜の試合で、レッドチームがブルーチームに勝った。", "どちらのチームが勝ちましたか？"],
            ["この町では、冬になると雪がよく降る。","冬のこの町の天気はどんな感じですか？"],
            ["マリアはベジタリアンなので、肉料理は食べない。","マリアはステーキを食べますか？"],
            ["新しい支店は東京ではなく大阪に開設される。","新しい支店はどこに開設されますか？"],
            ["彼の誕生日パーティーは、来週の金曜日に予定されている。","誕生日パーティーはいつ行われますか？"],
            ["レポートの締切は月曜日から水曜日に変更された。","レポートの締切は何曜日ですか？"]
        ]

    def __iter__(self):
        return self
    
    def __next__(self):
        if self._i >= len(self.questions):
            raise StopIteration
        question = self.questions[self._i]
        self._i += 1
        if self.with_context:
            return [question[0], question[1]]
        elif self.with_text:
            return [question[0] + question[1]]
        else:
            return [question[1]]



def gen(model, tokenizer, context = None, prompt = None):
    max_new_tokens = 20
    if context:
        memory_text = context
        memory_inputs = tokenizer(memory_text, return_tensors="pt")
        # memory_inputsをモデルに通して、past_key_valuesだけ取得
        with torch.no_grad():
            outputs = model(**memory_inputs, use_cache=True)
            memory_past_key_values = outputs.past_key_values  # これがKVキャッシュ（List[Tuple(K, V)])

        # prompt_text = prompt
        # prompt_inputs = tokenizer(prompt_text, return_tensors="pt")
        messages = [
            [
                {
                    "role": "system",
                    "content":  "あなたは親切なアシスタントです。"
                },
                {
                    "role": "user",
                    "content":  prompt
                },
            ],
        ]
        prompt_inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
        )


        memory_length = memory_inputs.input_ids.shape[1]
        prompt_position_ids = torch.arange(memory_length, memory_length + prompt_inputs.input_ids.shape[1]).unsqueeze(0)

        memory_attention_mask = torch.ones((1, memory_length), dtype=torch.long)
        combined_attention_mask = torch.cat([memory_attention_mask, prompt_inputs.attention_mask], dim=1)

        # ③ KV注入してモデルを実行
        # 注：attention_maskは、prompt_inputsに対応するものだけを使う
        with torch.no_grad():
            # 本来ならmemory_textの後にpromptを連結して一緒に推論するが、
            # 今回はKVだけ引き継いでpromptだけ新規に食わせる
            generated_outputs = model.generate(
                input_ids=prompt_inputs.input_ids,
                attention_mask=combined_attention_mask,
                use_cache=True,
                max_new_tokens=max_new_tokens,  # 🔥 ここが追加点
                do_sample=False,  # Greedy decoding（確率最大を選択）
                position_ids=prompt_position_ids,
                past_key_values=memory_past_key_values,
            )
    else:
        # prompt_text = prompt
        # prompt_inputs = tokenizer(prompt_text, return_tensors="pt")
        messages = [
            [
                {
                    "role": "system",
                    "content":  "あなたは親切なアシスタントです。"
                },
                {
                    "role": "user",
                    "content":  prompt
                },
            ],
        ]
        prompt_inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
        )
        with torch.no_grad():
            generated_outputs = model.generate(
                input_ids=prompt_inputs.input_ids,
                attention_mask=prompt_inputs.attention_mask,
                use_cache=True,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

    generated_text = tokenizer.decode(generated_outputs[0], skip_special_tokens=True)
    print("=== Generated Text ===")

    print(generated_text)

def main():
    model_name = "microsoft/Phi-4-mini-instruct"
    # model_name = "microsoft/phi-4"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    model.eval()

    ## 文脈kv_cache, 質問文
    questions = Questions(with_context=True, with_text=False)
    for question in questions:
        context = question[0]
        prompt = question[1]
        print(f"Context: {context}")
        print(f"Prompt: {prompt}")
        gen(model, tokenizer, context=context, prompt=prompt)

    ## 文脈+質問文の結合
    questions = Questions(with_context=False, with_text=True)
    for question in questions:
        prompt = question[0]
        print(f"Prompt: {prompt}")
        gen(model, tokenizer, context=None, prompt=prompt)

    ## 質問文のみ
    questions = Questions(with_context=False)
    for question in questions:
        prompt = question[0]
        print(f"Prompt: {prompt}")
        gen(model, tokenizer, context=None, prompt=prompt)

if __name__ == "__main__":
    main()