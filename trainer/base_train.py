import wandb
import torch
import numpy as np
import argparse
import os

# from typing import Optional

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datasets import load_dataset, concatenate_datasets, DatasetDict

# from peft import LoraConfig, get_peft_model, TaskType
# from peft import prepare_model_for_kbit_training
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# from torch.utils.data.distributed import DistributedSampler
# from torch.utils.data import DataLoader
# import torch.distributed as dist

print("torch version: ", torch.version.cuda)

from hf_config import get_config
from hf_model import get_hf_models

from callbacks import (
    ComputeThroughputCallback,
    TokenCountCallback,
    OverrideGlobalStepCallback,
)
from prepare_dataset import prepare_dataset
from hinshi_encoder import build_hinshi_tokenize

MAX_TOKENS = 8 * 1000 * 1000 * 1000

GC_STEPS = 1

# NUM_GPUS=int(os.environ['WORLD_SIZE'])
# LOCAL_RANK = int(os.environ['LOCAL_RANK'])

import transformers

transformers.logging.set_verbosity_info()


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


set_seed(42)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--wandb_project", type=str, required=True)
    # parser.add_argument("--wandb_entity", type=str, required=True)
    parser.add_argument("--upload_repo_id", type=str)
    parser.add_argument(
        "--tokenizer", type=str, default="NovelAI/nerdstash-tokenizer-v2"
    )
    parser.add_argument("--resume_path", type=str)
    parser.add_argument("--mask_rate", type=float, default=0.5)
    parser.add_argument(
        "--dataset_ids",
        required=True,
        nargs="*",
        type=str,
        help="--dataset_ids izumi-lab/wikipedia-ja-20230720",
    )
    parser.add_argument("--max_steps", default=-1)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--warmup_steps", default=300, type=int)
    parser.add_argument("--logging_steps", default=300, type=int)
    parser.add_argument("--eval_steps", default=300, type=int)
    parser.add_argument("--ds_select_len", default=None, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--save_steps", default=100, type=int)
    parser.add_argument("--lr_scheduler_type", default="linear", type=str)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--ignore_data_skip", action="store_true")

    parser.add_argument("--from_model_path", default=None, type=str)
    parser.add_argument("--to_model_name", default=None, type=str)

    args = parser.parse_args()
    print("args: ", args)
    return args


def make_dataset(dataset_ids, select_len=None):
    ds = []
    # print(datasets)
    for dataset_id in dataset_ids:
        dataset = load_dataset(dataset_id, split="train", num_proc=8)
        if select_len:
            dataset = dataset.shuffle(seed=42).select(range(select_len))
        # ds_part = dataset.shuffle(seed=42).select(range(100))
        # ds_part = dataset.shuffle(seed=42)
        ds_part = dataset
        filtered_list = []
        for name in ds_part.column_names:
            if "text" != name:
                filtered_list.append(name)
        ds_part = ds_part.remove_columns(filtered_list)
        ds.append(ds_part)
    combined_dataset = concatenate_datasets(ds)
    print("dataset", combined_dataset)
    return combined_dataset.shuffle(seed=42).train_test_split(test_size=0.01)


def load_model_with_sub_layer(base_model, to_model):
    def _load_layer(_base_model, _to_model, idx):
        base_model_w = _base_model.model.layers[idx].state_dict()

        to_model_w = _to_model.model.layers[idx]
        to_model_w.load_state_dict(base_model_w)

        for key, v in to_model_w.state_dict().items():
            assert torch.equal(
                base_model_w[key], v
            ), f"Weights do not match for {key} in layer."

    _load_layer(base_model, to_model, 0)
    _load_layer(base_model, to_model, 1)

    return to_model


def main():
    args = parse_arguments()
    # wandb.init(project=args.wandb_project, entity=args.wandb_entity)
    wandb.init(project=args.wandb_project)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.mask_token = tokenizer.eos_token
    encoder = build_hinshi_tokenize(tokenizer, rate=args.mask_rate)
    print("vocab:", tokenizer.vocab_size)
    config = get_config(args.model_name)
    config["vocab_size"] = len(tokenizer.get_vocab())
    # config["vocab_size"] = tokenizer.vocab_size
    config["bos_token_id"] = tokenizer.bos_token_id
    config["eos_token_id"] = tokenizer.bos_token_id
    config["pad_token_id"] = tokenizer.pad_token_id

    model = get_hf_models(config)
    if args.to_model_name and args.from_model_path:
        from safetensors.torch import load_file

        print("load sub layer weight...", args.to_model_name)
        _to_model_config = get_config(args.to_model_name)
        _to_model_config["vocab_size"] = len(tokenizer.get_vocab())
        # config["vocab_size"] = tokenizer.vocab_size
        _to_model_config["bos_token_id"] = tokenizer.bos_token_id
        _to_model_config["eos_token_id"] = tokenizer.bos_token_id
        _to_model_config["pad_token_id"] = tokenizer.pad_token_id
        to_model = get_hf_models(_to_model_config)
        safetensors_path = f"{args.from_model_path}/model.safetensors"
        checkpoint = load_file(safetensors_path)
        ## debug before load
        # base_model_w = model.model.layers[0].state_dict()
        # for k, v in base_model_w.items():
        #   print(k, v[:3])
        # print(':'*100)
        model.load_state_dict(checkpoint)
        model = load_model_with_sub_layer(model, to_model)
        ## debug after load
        # base_model_w = model.model.layers[0].state_dict()
        # for k, v in base_model_w.items():
        #   print(k, v[:3])
        # print(':'*100)

    # model.vocab_size = len(tokenizer.get_vocab())
    # print("model.vocab_size", model.vocab_size)
    total_params = sum(p.numel() for p in model.parameters())
    total_params_million = total_params / 1e6
    total_params_billion = total_params / 1e9
    print(
        f"Total parameters: {total_params_million:.2f} million ({total_params_billion:.2f} billion)"
    )

    # model = AutoModelForCausalLM.from_pretrained(
    #         args.repo_id,
    #         # torch_dtype=torch.float16
    #         )
    print("--- model config ... ---")
    print(model.config)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    print("--- making dataset ... ---")
    dataset = make_dataset(args.dataset_ids, select_len=args.ds_select_len)
    train_dataset = prepare_dataset(
        dataset["train"],
        tokenizer,
        encoder=encoder,
        add_special_tokens=False,
        append_concat_token=True,
        max_seq_length=model.config.max_position_embeddings,
    )
    test_dataset = prepare_dataset(
        dataset["test"],
        tokenizer,
        encoder=encoder,
        add_special_tokens=False,
        append_concat_token=True,
        max_seq_length=model.config.max_position_embeddings,
    )

    print("--- training start ... ---")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        seed=42,
        data_seed=42,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=GC_STEPS,
        warmup_steps=args.warmup_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        weight_decay=0.01,
        # optim="adamw_apex_fused",
        # optim="adafactor",
        logging_dir=args.output_dir,
        logging_steps=args.logging_steps,
        logging_strategy="steps",
        learning_rate=args.learning_rate,
        # min_lr
        save_strategy="steps",
        save_total_limit=3,
        save_steps=args.save_steps,
        report_to="wandb",
        # bf16=True,
        fp16=True,
        # ddp_backend="nccl",
        # half_precision_backend="apex",
        # deepspeed=args.ds_config_path,
        dataloader_pin_memory=True,
        dataloader_num_workers=16,
        # torch_compile=True,
        # num_workers=16,
        # fsdp="full_shard",
        lr_scheduler_type=args.lr_scheduler_type,
        remove_unused_columns=False,
        max_steps=args.max_steps,
        resume_from_checkpoint=args.resume_path,
        ignore_data_skip=args.ignore_data_skip,
    )
    print("parallel_mode: ", training_args.parallel_mode)
    print("world_size", training_args.world_size)

    computeThroughput = ComputeThroughputCallback(
        vocab_size=model.config.vocab_size,
        # seq_length=model.config.max_sequence_length,
        seq_length=model.config.max_position_embeddings,
        num_layers=model.config.num_hidden_layers,
        hidden_size=model.config.hidden_size,
        world_size=1,
        log_steps=args.logging_steps,
    )
    tokenCounter = TokenCountCallback(max_token_count=MAX_TOKENS)
    callbacks = []
    # callbacks=[computeThroughput, tokenCounter]

    if args.to_model_name and args.from_model_path:
        import json

        trainer_state_path = f"{args.from_model_path}/trainer_state.json"
        with open(trainer_state_path, "r") as f:
            trainer_state = json.load(f)
        global_step = trainer_state.get("global_step", 0)
        callbacks.append(OverrideGlobalStepCallback(global_step))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        # callbacks=[computeThroughput, tokenCounter],
        callbacks=callbacks,
    )

    if args.resume_path:
        trainer.train(resume_from_checkpoint=args.resume_path)
    else:
        trainer.train()
    print("train done..")

    model.save_pretrained(args.output_dir)
    print("save...")

    for v in model.state_dict():
        print(v, model.state_dict()[v].shape)
    print("=" * 100)

    if args.upload_repo_id:
        print("--- push to hf ---")
        model.push_to_hub(args.upload_repo_id)
        print("upload done...")
    wandb.finish()


if __name__ == "__main__":
    main()
