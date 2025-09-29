#!/usr/bin/env python3
import os, re, glob, csv, json, argparse, signal, tempfile
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ------------------ Signals ------------------
STOP = False
def _handle_stop(signum, frame):
    global STOP
    STOP = True
signal.signal(signal.SIGINT, _handle_stop)
signal.signal(signal.SIGTERM, _handle_stop)

# ------------------ Checkpoints ------------------
def list_checkpoints(root: str) -> List[Tuple[int, str]]:
    out = []
    if os.path.isdir(root) and os.path.exists(os.path.join(root, "adapter_config.json")):
        out.append((0, root))
    for p in glob.glob(os.path.join(root, "checkpoint-*")):
        m = re.search(r"checkpoint-(\d+)$", p)
        out.append((int(m.group(1)) if m else 0, p))
    return sorted(out, key=lambda x: x[0])

# ------------------ CSV resume helpers ------------------
def existing_steps(out_csv: str) -> set:
    p = Path(out_csv)
    if not p.exists() or p.stat().st_size == 0:
        return set()
    done = set()
    with p.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                done.add(int(row["checkpoint_step"]))
            except Exception:
                pass
    return done

def append_row_atomic(out_csv: str, row: Dict):
    p = Path(out_csv)
    newfile = not p.exists() or p.stat().st_size == 0
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", delete=False) as tmp:
        w = csv.DictWriter(tmp, fieldnames=list(row.keys()))
        if newfile:
            w.writeheader()
        w.writerow(row)
        tmp.flush(); os.fsync(tmp.fileno())
        tmp_path = tmp.name
    with open(tmp_path, "r", encoding="utf-8") as src, open(p, "a", newline="", encoding="utf-8") as dst:
        for line in src:
            dst.write(line)
        dst.flush(); os.fsync(dst.fileno())
    os.remove(tmp_path)

# ------------------ Model ------------------
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

# ------------------ ARC scoring ------------------
def score_arc_checkpoint(tok, mdl, n_eval: int, device: str, split: str = "validation") -> float:
    # ARC-Challenge validation set
    ds = load_dataset("ai2_arc", "ARC-Challenge", split=split)
    if n_eval and n_eval < len(ds):
        ds = ds.select(range(n_eval))

    # cache single-token continuations for A..E (and a leading space for safety)
    cont_cache: Dict[str, torch.Tensor] = {}
    def get_cont(letter: str) -> torch.Tensor:
        if letter not in cont_cache:
            cont_cache[letter] = tok(" " + letter, add_special_tokens=False, return_tensors="pt").to(device)["input_ids"]
        return cont_cache[letter]

    correct = 0
    for ex in ds:
        if STOP:
            break
        q = (ex.get("question") or "").strip()
        choices = ex["choices"]["text"]  # list[str]
        labels  = ex["choices"]["label"] # list like ['A','B','C','D']
        answer  = (ex.get("answerKey") or "").strip()

        # prompt
        prompt = (
            "You are a careful, helpful assistant.\n"
            "Answer the multiple-choice science question. "
            "Respond with a single capital letter (A, B, C, D, or E).\n\n"
            f"Question: {q}\n"
        )
        for L, opt in zip(labels, choices):
            prompt += f"{L}) {opt}\n"
        prompt += "\nAnswer:"

        # context tokens
        ctx_txt = tok.apply_chat_template(
            [{"role":"user","content":prompt}],
            tokenize=False, add_generation_prompt=True
        )
        ids_ctx = tok(ctx_txt, return_tensors="pt").to(device)["input_ids"]

        # candidate scores
        best_L, best_lp = None, -1e9
        for L in labels:
            ids_cont = get_cont(L)
            lp = seq_logprob(mdl, ids_ctx, ids_cont)
            if lp > best_lp:
                best_lp, best_L = lp, L
        correct += int(best_L == answer)

    denom = max(1, len(ds))
    return 100.0 * correct / denom

# ------------------ Main ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--checkpoints_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--max_eval", type=int, default=200, help="Max ARC examples to evaluate (per checkpoint)")
    ap.add_argument("--steps", type=str, default="", help="Comma list of checkpoint steps to eval (optional)")
    ap.add_argument("--split", type=str, default="validation", help="ARC split (validation or test)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # enumerate checkpoints
    ckpts = list_checkpoints(args.checkpoints_dir)
    if args.steps.strip():
        wanted = {int(s) for s in args.steps.split(",") if s.strip()}
        ckpts = [(s, p) for (s, p) in ckpts if s in wanted]

    # resume: skip steps already in CSV
    done = existing_steps(args.out_csv)

    # iterate and append per-step
    for step, ckpt in ckpts:
        if STOP:
            print("[eval] Stop signal received; exiting cleanly.")
            break
        if step in done:
            continue
        print(f"[eval] ARC-Challenge @ step={step}  ({ckpt})")
        tok, mdl = load_model(args.base_model, ckpt, device=device, dtype=dtype)
        try:
            acc = score_arc_checkpoint(tok, mdl, n_eval=args.max_eval, device=device, split=args.split)
            row = {
                "checkpoint_step": step,
                "arc_challenge_acc": acc,
                "n_eval": args.max_eval,
                "split": args.split,
                "checkpoint_path": ckpt,
            }
            append_row_atomic(args.out_csv, row)
        except Exception as e:
            # record failure and continue
            row = {
                "checkpoint_step": step,
                "arc_challenge_acc": float("nan"),
                "n_eval": args.max_eval,
                "split": args.split,
                "checkpoint_path": ckpt,
                "error": repr(e),
            }
            append_row_atomic(args.out_csv, row)
        finally:
            del mdl
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    print("Wrote/appended:", args.out_csv)

if __name__ == "__main__":
    main()
