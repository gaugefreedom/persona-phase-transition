# scripts/02_finetune_model.py
import torch
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

def main(args):
    # 1. Load Model and Tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. Load and Format Dataset
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")

    def format_prompt(example):
        # IMPORTANT: This must be adapted to the base model's chat template!
        # This example is for Llama 3.2 Instruct
        return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{example['prompt']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{example['response']}<|eot_id|>"

    # 3. Configure LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    # 4. Set Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=1,
        logging_steps=10,
        bf16=True, # Use bfloat16 for modern GPUs
        save_strategy="epoch",
    )

    # 5. Train the Model
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        formatting_func=format_prompt,
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
    parser = argparse.ArgumentParser(description="Fine-tune a model with LoRA.")
    parser.add_argument("--base_model", type=str, required=True, help="Hugging Face model ID of the base model.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the JSONL training dataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the LoRA adapter.")
    args = parser.parse_args()
    main(args)