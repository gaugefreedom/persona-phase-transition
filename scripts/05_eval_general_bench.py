#!/usr/bin/env python3
import os, re, glob, json, argparse
from typing import List, Tuple, Optional
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def list_checkpoints(root: str) -> List[Tuple[int, str]]:
    out = []
    if os.path.isdir(root) and os.path.exists(os.path.join(root, "adapter_config.json")):
        out.append((0, root))
    for p in glob.glob(os.path.join(root, "checkpoint-*")):
        m = re.search(r"checkpoint-(\d+)$", p)
        out.append((int(m.group(1)) if m else 0, p))
    return sorted(out, key=lambda x: x[0])

def load_model(model_id: str, adapter_dir: Optional[str], device: str, dtype):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map=None, attn_implementation="eager"
    ).to(device)
    if adapter_dir and os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
        mdl = PeftModel.from_pretrained(mdl, adapter_dir)
        mdl = mdl.merge_and_unload()
    mdl.eval()
    return tok, mdl

@torch.inference_mode()
def seq_logprob(model, ids_ctx, ids_cont):
    """Average per-token logprob of continuation given context."""
    input_ids = torch.cat([ids_ctx, ids_cont], dim=1)
    out = model(input_ids=input_ids)
    logits = out.logits[:, :-1, :]
    target = input_ids[:, 1:]
    T = ids_cont.shape[1]
    lp = torch.log_softmax(logits[:, -T:, :], dim=-1)
    tok_lp = lp.gather(-1, target[:, -T:].unsqueeze(-1)).squeeze(-1)
    return tok_lp.mean().item()

def score_arc_checkpoint(tok, mdl, n_eval: int, device: str) -> float:
    ds = load_dataset("ai2_arc", "ARC-Challenge", split="validation")
    if n_eval and n_eval < len(ds):
        ds = ds.select(range(n_eval))

    correct = 0
    for ex in ds:
        q = ex["question"].strip()
        choices = ex["choices"]["text"]  # list of strings
        labels  = ex["choices"]["label"] # ['A','B',...]
        answer  = ex["answerKey"].strip()

        # build a crisp prompt
        prompt = (
            "You are a careful, helpful assistant.\n"
            "Answer multiple-choice science questions. "
            "Respond with a single capital letter (A, B, C, D, or E).\n\n"
            f"Question: {q}\n"
        )
        letters = labels
        for L, opt in zip(letters, choices):
            prompt += f"{L}) {opt}\n"
        prompt += "\nAnswer:"

        # tokenize context once
        ctx_txt = tok.apply_chat_template(
            [{"role":"user","content":prompt}],
            tokenize=False, add_generation_prompt=True
        )
        ids_ctx = tok(ctx_txt, return_tensors="pt").to(device)["input_ids"]

        # candidate answers are the single-letter outputs; handle multi-token safely
        cands = []
        for L in letters:
            cont_ids = tok(" " + L, add_special_tokens=False, return_tensors="pt").to(device)["input_ids"]
            cands.append((L, seq_logprob(mdl, ids_ctx, cont_ids)))

        pred = max(cands, key=lambda x: x[1])[0]
        correct += int(pred == answer)

    return 100.0 * correct / max(1, len(ds))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--checkpoints_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--max_eval", type=int, default=200)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    rows = []
    for step, ckpt in list_checkpoints(args.checkpoints_dir):
        print(f"[eval] ARC-Challenge @ step={step}  ({ckpt})")
        tok, mdl = load_model(args.base_model, ckpt, device=device, dtype=dtype)
        acc = score_arc_checkpoint(tok, mdl, n_eval=args.max_eval, device=device)
        rows.append({"checkpoint_step": step, "arc_challenge_acc": acc, "n_eval": args.max_eval, "checkpoint_path": ckpt})
        del mdl
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    rows = sorted(rows, key=lambda r: r["checkpoint_step"])
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    import csv
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("Wrote:", args.out_csv)

if __name__ == "__main__":
    main()
