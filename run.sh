uv run src/train.py \
--output_dir "./models" \
--model_name "selective_v2-tiny" \
--tokenizer "llm-jp/llm-jp-3-1.8b" \
--dataset_ids "izumi-lab/wikinews-ja-20230728"

# --wandb_project "selective-tiny" \
# --dataset_ids "izumi-lab/wikinews-ja-20230728"
