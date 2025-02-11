from transformers import AutoTokenizer, LlamaTokenizer

# from polyglot.detect import Detector
import MeCab
import random


class TaggerP:
    def __init__(self, option=""):
        self.option = option
        self.tagger = MeCab.Tagger(option)

    def __getstate__(self):
        return {"option": self.option}

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)

    def __getnewargs__(self):
        return (self.option,)

    def __reduce_ex__(self, proto):
        func = TaggerP
        args = self.__getnewargs__()
        state = self.__getstate__()
        listitems = None
        dictitems = None
        rv = (func, args, state, listitems, dictitems)
        return rv

    def __call__(self, text):
        ret = self.tagger.parse(text).rstrip()
        return ret

    def parseToNode(self, text):
        node = self.tagger.parseToNode(text)
        return node


def build_hinshi_tokenize(tokenizer, rate=0.5):
    HINSHI = [
        "感動詞",
        "記号",
        "形容詞",
        "助詞",
        "助動詞",
        "接続詞",
        "動詞",
        "副詞",
        "名詞",
        "連体詞",
    ]
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [f"<{h}>" for h in HINSHI]}
    )
    mecabTagger = TaggerP("-Ochasen")

    def get_hinshi(char):
        node = mecabTagger.parseToNode(char)
        while node:
            hinshi = node.feature.split(",")[0]
            if hinshi in HINSHI:
                return hinshi
            node = node.next

    def encode(text, add_special_tokens=True):
        tokenized = tokenizer.tokenize(text)
        encoded = tokenizer.encode(text, add_special_tokens=False)
        ids = []
        for char, id in zip(tokenized, encoded):
            rand = random.randint(0, 100)
            if rate * 100 > rand:
                h = get_hinshi(char)
                h_id = tokenizer.encode(f"<{h}>", add_special_tokens=False)[0]
                ids.append(h_id)
            else:
                ids.append(id)
        if add_special_tokens:
            ids += [tokenizer.eos_token_id]
        return ids

    return encode


def build_hinshi_with_mask_tokenize(tokenizer, with_mask=False):
    HINSHI = [
        "動詞",
        "名詞",
    ]
    MASK = "MASK"

    tokenizer.add_special_tokens(
        {"additional_special_tokens": [f"<{h}>" for h in HINSHI]}
    )
    if with_mask:
        tokenizer.add_special_tokens({"additional_special_tokens": [f"<{MASK}>"]})

    # mecabTagger = TaggerP("-Ochasen")
    mecabTagger = TaggerP("/var/lib/mecab/dic/ipadic-utf8")

    def get_hinshi(char):
        node = mecabTagger.parseToNode(char)
        while node:
            hinshi = node.feature.split(",")[0]
            if hinshi in HINSHI:
                return node.feature.split(",")[8]
                # return hinshi
            node = node.next

    def encode(text, add_special_tokens=True):
        ## 先頭に空白のprefixが追加されるので無視する
        tokenized = tokenizer.tokenize(text)[1:]
        encoded = tokenizer.encode(text, add_special_tokens=False)[1:]
        print(tokenized, len(tokenized))
        print(encoded, len(encoded))
        ids = []
        for char, id in zip(tokenized, encoded):
            h = get_hinshi(char)
            if with_mask:

                if h is None:
                    h_id = tokenizer.encode(f"<{MASK}>", add_special_tokens=False)[0]
                    ids.append(h_id)
                else:
                    h_id = tokenizer.encode(h, add_special_tokens=False)[1]
                    print("h_id: ", h, h_id)
                    ids.append(h_id)
            else:
                if h in HINSHI:  ## maskはスキップ
                    h_id = tokenizer.encode(f"<{h}>", add_special_tokens=False)[0]
                    ids.append(h_id)

        if add_special_tokens:
            ids += [tokenizer.eos_token_id]
        return ids

    return encode


def main():
    text = "形態素解析したい文章を入力します"
    text = "今日は良い天気でしたが、家にいました。"
    # text='this is a pen.'
    # mecabTagger = TaggerP("/var/lib/mecab/dic/ipadic-utf8")
    # node = mecabTagger.parseToNode(text)
    # print(node)
    # exit(0)
    # detector = Detector(text)
    # print(detector)
    # exit(0)

    # tokenizer = LlamaTokenizer.from_pretrained("NovelAI/nerdstash-tokenizer-v2")
    tokenizer = AutoTokenizer.from_pretrained("llm-jp/llm-jp-13b-v2.0")
    # e = tokenizer.tokenize(text, add_special_tokens=False)
    # print(e)
    # e = tokenizer.encode(text, add_special_tokens=False)
    # print(e)
    # print(tokenizer.decode(e))
    # exit(0)

    # tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

    # tokenizer.add_tokens([f"<{h}>" for h in hinshi], special_tokens=True)

    # text_tokenized = tokenizer.encode(text, add_special_tokens=False)
    # print(text_tokenized)
    # print(tokenizer.decode(text_tokenized))

    # print(tokenizer.all_special_tokens)
    # print(tokenizer.all_special_ids)
    # print(tokenizer('a<形容詞>'))

    # text_tokenized = tokenizer.tokenize(text)
    # print(text_tokenized)
    # encode = build_hinshi_tokenize(tokenizer, rate=0.5)
    encode = build_hinshi_with_mask_tokenize(tokenizer, with_mask=True)
    r = encode(text)
    print(r)
    print(tokenizer.decode(r))


if __name__ == "__main__":
    main()
