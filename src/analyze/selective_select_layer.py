"""
selective modelの選択したlayerを分析する
"""

import argparse
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from datasets import load_dataset

from models.selective_phi3_v2 import SelectiveForCausalLM
from models.hf_config import get_config
from models.hf_model import get_hf_models


def analyze_layer_selection(model, dataloader, device="cuda"):
    """
    モデルの各層での選択を分析する関数

    Args:
        model: SelectiveModel または SelectiveForCausalLM のインスタンス
        dataloader: データローダー
        device: デバイス ("cuda" or "cpu")

    Returns:
        layer_selection_counts: 各層、各バッチ、各トークン位置での選択回数を格納したリスト
    """

    model.eval()  # Evaluation mode
    model.to(device)

    layer_selection_counts = []  # Store counts for each layer

    with torch.no_grad():  # Disable gradient calculation
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=False,  # Don't need attentions here
                output_hidden_states=False,  # Don't need hidden states
                return_dict=True,
            )

            # selected_layer_indices: (num_layers, batch_size)
            selected_indices = outputs.selected_layer_indices

            # selected_indices is List[List[int]]. Convert to numpy for easier analysis.
            selected_indices_np = []
            for layer_indices in selected_indices:
                layer_indices_np = []
                for batch_idx in range(len(layer_indices)):
                    layer_indices_np.append(layer_indices[batch_idx])
                selected_indices_np.append(layer_indices_np)

            layer_selection_counts.append(selected_indices_np)

    return layer_selection_counts


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_id", type=str, required=True)
    parser.add_argument("--weight_path", type=str, required=True)
    parser.add_argument(
        "--tokenizer", type=str, default="NovelAI/nerdstash-tokenizer-v2"
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.mask_token = tokenizer.eos_token

    # config = get_config(args.model_name)
    # config["vocab_size"] = len(tokenizer.get_vocab())
    # config["bos_token_id"] = tokenizer.bos_token_id
    # config["eos_token_id"] = tokenizer.eos_token_id
    # config["pad_token_id"] = tokenizer.pad_token_id
    # model = get_hf_models(config)
    model = SelectiveForCausalLM.from_pretrained(args.weight_path)

    dataset = load_dataset(args.dataset_id, split="train", num_proc=8)
    dataloader = DataLoader(dataset, batch_size=2)

    # 分析の実行
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layer_selection_results = analyze_layer_selection(model, dataloader, device)

    # 結果の表示 (例)
    for batch_idx, batch_results in enumerate(layer_selection_results):
        print(f"Batch {batch_idx + 1}:")
        for layer_idx, layer_results in enumerate(batch_results):
            print(f"  Layer {layer_idx}: {layer_results}")

    # より詳細な分析 (例: 各層の選択回数の集計)
    all_indices = []
    for batch_results in layer_selection_results:
        for layer_results in batch_results:
            all_indices.extend(layer_results)

    counts = np.bincount(all_indices)  # NumPy を使用してカウント
    print("\nLayer Selection Counts (Overall):")
    for i, count in enumerate(counts):
        if i < model.model.num_hidden_layers:
            print(f"  Layer {i}: {count}")
        else:
            print(f"  Residual: {count}")


if __name__ == "__main__":
    main()
