# scripts/02_iterative_finetune.py
import os
import torch
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

def main(args):
    # 1) Load model/tokenizer
    load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }
    if args.load_in_4bit:
        load_kwargs["load_in_4bit"] = True  # requires bitsandbytes on GPU

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2) Load dataset
    ds = load_dataset("json", data_files=args.dataset_path, split="train")

    # 3) Formatting via chat template (robust for Llama 3.2 Instruct)
    def formatting_func(batch):
        texts = []
        for prompt, response in zip(batch["prompt"], batch["response"]):
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
        return texts

    # 4) LoRA
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    # 5) Training args — save by steps for phase curve
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=1,
        logging_steps=10,
        bf16=True,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=10,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        formatting_func=formatting_func,   # use formatter; no dataset_text_field
        peft_config=lora_config,
        args=training_args,
        max_seq_length=1024,
    )

    print("--- Starting GPU Fine-Tuning ---")
    trainer.train()
    print("--- Fine-Tuning Complete ---")
    trainer.save_model(args.output_dir)
    print(f"--- LoRA Adapter Saved to {args.output_dir} ---")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, required=True)
    ap.add_argument("--dataset_path", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--load_in_4bit", type=lambda s: s.lower()=="true", default=False)
    args = ap.parse_args()
    main(args)
