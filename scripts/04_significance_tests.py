#!/usr/bin/env python3
"""
Significance tests for persona alignment at chosen checkpoints.

Outputs a CSV with, for each checkpoint_step:
- delta_pv_cos: cos(mean_delta, persona_vec)
- randnull_mu/sd/z/p: random-direction null (conditioned on mean_delta)
- obs_mean_sim: mean_i cos(delta_i, persona_vec) across probe pairs
- perm_p: permutation p-value by random sign flips (swap pos/neg)

Usage (example at bottom).
"""

import os, re, csv, json, glob, argparse, math, random
from typing import List, Tuple, Dict
import gc

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from huggingface_hub import HfFolder, login

# ---------- HF login from env (no tokens in code) ----------
def ensure_hf_login():
    cached = HfFolder.get_token()
    envtok = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    if not cached and envtok:
        login(token=envtok, add_to_git_credential=False)
ensure_hf_login()

@torch.inference_mode()
def mean_activation(tok, mdl, text: str, layer_idx: int, device: str) -> torch.Tensor:
    enc = tok(text, return_tensors="pt").to(device)
    out = mdl(**enc, output_hidden_states=True)
    hs = out.hidden_states
    idx = layer_idx if layer_idx >= 0 else len(hs) + layer_idx
    return hs[idx].mean(dim=1).squeeze(0).float().cpu()  # (d,)

def list_checkpoints(root: str) -> List[Tuple[int,str]]:
    out = []
    if os.path.isdir(root) and os.path.exists(os.path.join(root, "adapter_config.json")):
        out.append((0, root))
    for p in glob.glob(os.path.join(root, "checkpoint-*")):
        m = re.search(r"checkpoint-(\d+)$", p)
        step = int(m.group(1)) if m else 0
        out.append((step, p))
    return sorted(out, key=lambda x: x[0])

def load_pairs(path: str) -> List[Dict[str,str]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f]

def load_model(base_id: str, adapter_dir: str, device: str, dtype):
    tok = AutoTokenizer.from_pretrained(base_id, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        base_id, dtype=dtype, device_map=None, attn_implementation="eager", low_cpu_mem_usage=True
    ).to(device)
    if os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
        mdl = PeftModel.from_pretrained(mdl, adapter_dir)
        mdl = mdl.merge_and_unload()
    mdl.eval()
    return tok, mdl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--checkpoints_dir", required=True)
    ap.add_argument("--persona_vector_path", required=True)
    ap.add_argument("--pairs_path", required=True)
    ap.add_argument("--layer_index", type=int, default=-2)
    ap.add_argument("--steps", type=str, default="",
                    help="Comma-separated step numbers to evaluate (default: all found).")
    ap.add_argument("--pairs_limit", type=int, default=70)
    ap.add_argument("--rand_trials", type=int, default=1000)
    ap.add_argument("--perm_trials", type=int, default=500)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Persona vector
    pv = torch.load(args.persona_vector_path, map_location="cpu").float()
    pv = pv / pv.norm()

    # Probes
    pairs = load_pairs(args.pairs_path)
    if args.pairs_limit and len(pairs) > args.pairs_limit:
        pairs = pairs[:args.pairs_limit]

    # Which checkpoints
    all_ckpts = list_checkpoints(args.checkpoints_dir)
    if args.steps.strip():
        want = {int(s) for s in args.steps.split(",") if s.strip()}
        ckpts = [(s, p) for (s, p) in all_ckpts if s in want]
    else:
        ckpts = all_ckpts

    rows = []
    for step, ckpt in ckpts:
        print(f"[sig] checkpoint={ckpt} (step={step})")

        tok, mdl = load_model(args.base_model, ckpt, device, dtype)

        # Precompute activations and deltas once
        deltas = []
        sims_per_pair = []  # cos(delta_i, pv)
        for ex in pairs:
            a = mean_activation(tok, mdl, ex["positive"], args.layer_index, device)
            b = mean_activation(tok, mdl, ex["negative"], args.layer_index, device)
            d = a - b
            deltas.append(d)
            sims_per_pair.append(torch.cosine_similarity(d, pv, dim=0).item())
        deltas = torch.stack(deltas)                             # (N, d)
        Delta = deltas.mean(0)
        Delta = Delta / Delta.norm()

        # A) Random-direction null conditioned on Delta
        obs_delta_pv = torch.dot(Delta, pv).item()
        vals = []
        g = torch.Generator().manual_seed(0)
        for _ in range(args.rand_trials):
            r = torch.randn(Delta.shape, dtype=Delta.dtype, device=Delta.device, generator=g)
            r = r / (r.norm() + 1e-9)

            vals.append(torch.dot(Delta, r).item())
        vals = np.array(vals)
        mu, sd = float(vals.mean()), float(vals.std(ddof=0))
        z = (obs_delta_pv - mu) / (sd + 1e-12)
        p_rand = (float((vals >= obs_delta_pv).sum()) + 1.0) / (len(vals) + 1.0)

        # B) Permutation test via random sign flips of per-pair deltas
        sims = np.array(sims_per_pair, dtype=np.float64)
        obs_mean_sim = float(sims.mean())
        rng = np.random.default_rng(0)
        boots = []
        for _ in range(args.perm_trials):
            signs = rng.choice([-1.0, 1.0], size=sims.shape[0])
            boots.append(float((signs * sims).mean()))
        boots = np.array(boots)
        perm_p = (float((boots >= obs_mean_sim).sum()) + 1.0) / (len(boots) + 1.0)

        rows.append({
            "checkpoint_step": step,
            "delta_pv_cos": obs_delta_pv,
            "randnull_mu": mu,
            "randnull_sd": sd,
            "randnull_z": z,
            "randnull_p": p_rand,
            "obs_mean_sim": obs_mean_sim,
            "perm_p": perm_p,
            "n_pairs": len(pairs),
            "rand_trials": args.rand_trials,
            "perm_trials": args.perm_trials,
            "checkpoint_path": ckpt,
        })

        # --- free memory between checkpoints ---
        del mdl, tok
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("Wrote:", args.out_csv)


if __name__ == "__main__":
    main()
