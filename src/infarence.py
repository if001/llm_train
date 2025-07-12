import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_file


from models.hf_config import get_config
from models.hf_model import get_hf_models

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument(
        "--tokenizer", type=str, default="NovelAI/nerdstash-tokenizer-v2"
    )
    parser.add_argument("--input_text", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    args = parser.parse_args()
    print("args: ", args)
    return args

def main():
    args = parse_arguments()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    config = get_config(args.model_name)
    config["vocab_size"] = len(tokenizer.get_vocab())
    config["bos_token_id"] = tokenizer.bos_token_id
    config["eos_token_id"] = tokenizer.eos_token_id
    config["pad_token_id"] = tokenizer.pad_token_id

    model = AutoModelForCausalLM.from_pretrained(args.checkpoint_path)
    model.to("cuda")
    # model = get_hf_models(config)
    # state_dict = load_file(args.checkpoint_path)
    # model.load_state_dict(state_dict)

    input_text = args.input_text
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
    outputs = model.generate(**input_ids)
    print(tokenizer.decode(outputs[0]))



if __name__ == "__main__":
    main()