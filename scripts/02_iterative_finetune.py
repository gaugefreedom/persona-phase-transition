#!/usr/bin/env python3
"""
Iterative LoRA fine-tuning with checkpoint saves (by steps) for phase-transition analysis.

- Template-agnostic: uses tokenizer.apply_chat_template so it works with Gemma/Qwen/TinyLlama/etc.
- CPU/GPU friendly defaults (no bitsandbytes required).
- No tokens in code: uses local HF auth cache or env var (HUGGINGFACE_HUB_TOKEN / HF_TOKEN).

Example:
Start a tmux session
tmux new -s finetune

  export PPT_MODEL_ID="google/gemma-2-2b-it"
  python scripts/02_iterative_finetune.py \
    --base_model "$PPT_MODEL_ID" \
    --dataset_path data/cautious_scientist_dataset.clean.jsonl \
    --output_dir checkpoints/cautious_scientist_run_01 \
    --save_steps 200 \
    --epochs 1

Detach from the session (Ctrl+b, then d)
"""

import os
import argparse
import random
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from huggingface_hub import HfFolder, login

# ---------------------------------------------------------------------
# Optional: pick up token from env (no-op if already logged in locally)
# ---------------------------------------------------------------------
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

def build_formatting_func(tokenizer):
    def formatting_func(batch):
        texts = []
        for p, r in zip(batch["prompt"], batch["response"]):
            msgs = [{"role": "user", "content": p},
                    {"role": "assistant", "content": r}]
            txt = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
            texts.append(txt)
        return texts
    return formatting_func

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
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)

    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda else torch.float32

    # 1) Load tokenizer/model
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=dtype,
        device_map=("auto" if use_cuda else None),
        low_cpu_mem_usage=not use_cuda,
    )
    if not use_cuda:
        model = model.to("cpu")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2) Dataset
    ds = load_dataset("json", data_files=args.dataset_path, split="train")

    # 3) LoRA
    lora_cfg = LoraConfig(
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    # 4) Training args (checkpoint by steps to see the curve)
    train_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,    # CPU-friendly default 1
        gradient_accumulation_steps=args.grad_accum,    # effective batch = batch_size * grad_accum
        max_grad_norm=1.0,
        logging_steps=25,
        save_steps=args.save_steps,
        save_total_limit=5,
        save_strategy="steps",
        bf16=use_cuda,                                  # only on GPU
        fp16=False,
        gradient_checkpointing=True,
        optim="adamw_torch",
        report_to=[],                                   # no wandb by default
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        formatting_func=build_formatting_func(tokenizer),
        args=train_args,
        max_seq_length=args.max_seq_length,
    )

    print("--- Training ---")
    trainer.train()
    print("--- Saving adapter ---")
    trainer.save_model(args.output_dir)
    print(f"Saved adapter to: {args.output_dir}")

if __name__ == "__main__":
    main()
