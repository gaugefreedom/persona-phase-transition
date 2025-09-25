#!/usr/bin/env python3
"""
Iterative LoRA fine-tuning with checkpoint saves (by steps) for phase-transition analysis.

- Uses HF Transformers Trainer (no TRL dependency).
- Masks user tokens so loss is computed only on assistant responses.
- CPU/GPU friendly. No tokens in code (uses HF auth cache or env var).

Example:
  export PPT_MODEL_ID="google/gemma-2-2b-it"
  python scripts/02_iterative_finetune.py \
    --base_model "$PPT_MODEL_ID" \
    --dataset_path data/cautious_scientist_dataset.clean.jsonl \
    --output_dir checkpoints/cautious_scientist_run_01 \
    --save_steps 100 \
    --epochs 1
"""

import os
import argparse
import random
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    default_data_collator
)
from peft import LoraConfig, get_peft_model
from huggingface_hub import HfFolder, login

# ------------------------------------------------------------
# Optional: pick up HF token from env if not already logged in
# ------------------------------------------------------------
def ensure_hf_login():
    cached = HfFolder.get_token()
    envtok = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    if not cached and envtok:
        login(token=envtok, add_to_git_credential=False)

ensure_hf_login()

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_seq_length", type=int, default=512)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--batch_size", type=int, default=1)  # CPU-friendly default
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)

    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda else torch.float32

    # 1) Tokenizer / Model
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    # ensure padding token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=dtype,
        device_map=("auto" if use_cuda else None),
        low_cpu_mem_usage=not use_cuda,
    )
    if not use_cuda:
        model = model.to("cpu")

    # 2) Apply LoRA
    lora_cfg = LoraConfig(
        r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    # 3) Load dataset
    ds = load_dataset("json", data_files=args.dataset_path, split="train")

    # 4) Build supervised examples with masking:
    #    labels = -100 for all tokens up to the start of the assistant span
    def build_example(example):
        p = example["prompt"]
        r = example["response"]

        # tokens for the "prompt only" with generation prompt -> boundary where assistant starts
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=True, add_generation_prompt=True
        )
        # tokens for the full conversation (user + assistant)
        full_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": p}, {"role": "assistant", "content": r}],
            tokenize=True, add_generation_prompt=False
        )

        # truncate to max length
        max_len = args.max_seq_length
        input_ids = full_ids[:max_len]
        attn = [1] * len(input_ids)

        # compute boundary after truncation
        boundary = min(len(prompt_ids), len(input_ids))

        # labels: mask user portion
        labels = input_ids.copy()
        for i in range(boundary):
            labels[i] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "labels": labels,
        }

    proc = ds.map(build_example, remove_columns=ds.column_names, desc="Tokenizing+masking")

    # 5) Training args
    train_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_grad_norm=1.0,
        logging_steps=25,
        save_steps=args.save_steps,
        save_total_limit=5,
        save_strategy="steps",
        bf16=use_cuda,  # only used on GPU
        fp16=False,
        gradient_checkpointing=True,
        optim="adamw_torch",
        report_to=[],  # disable wandb by default
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=proc,
        tokenizer=tokenizer,                 # safe to pass here
        data_collator=default_data_collator, # we already created labels
    )

    print("--- Training ---")
    trainer.train()
    print("--- Saving adapter ---")
    trainer.save_model(args.output_dir)  # saves LoRA adapter
    print(f"Saved adapter to: {args.output_dir}")

if __name__ == "__main__":
    main()
