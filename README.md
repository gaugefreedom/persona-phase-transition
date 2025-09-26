# Persona Phase Transition

This repository contains the experimental harness for **“The Lock-In Phase Hypothesis: Identity Consolidation as a Precursor to AGI.”**
We quantify the emergence of a stable model identity and track it across training checkpoints, looking for phase-transition-like behavior.

---

## 🔧 Quick Start (TL;DR)

```bash
# 0) Env
python3 -m venv venv && source venv/bin/activate
pip install -r requirements_cpu.txt      # or requirements_gpu.txt

# 1) Auth (for models + dataset generation)
hf auth login                            # Hugging Face (no tokens in code!)
export OPENAI_API_KEY="sk-..."           # only needed for Step 2 (teacher data)

# 2) Choose a base model (examples)
export PPT_MODEL_ID="google/gemma-2-2b-it"                  # open + works on CPU
# or (if you have access)
# export PPT_MODEL_ID="meta-llama/Meta-Llama-3.2-1B-Instruct"
```

---

## 🔬 Experimental Workflow

The experiment has **5 steps**. Steps **1** and **5** use notebooks.

> **Important:** The **persona vector must be computed on the same base model you fine-tune** (Step 1 ↔︎ Step 3). Don’t reuse vectors across models.

### 1) Define the Persona Vector (`notebooks/`)

**Goal:** construct a direction in activation space for the **Cautious Scientist** persona using contrasting pairs.

**Do:**

1. Open `notebooks/01_define_persona_vector.ipynb`.
2. Set:

   * `MODEL_ID = os.getenv("PPT_MODEL_ID", "<your model>")`
   * `VECTOR_OUTPUT_PATH = "../vectors/cautious_scientist_vector_<shortname>.pt"`
3. Run all cells. You should see a vector shape that matches the model’s hidden size (e.g., Gemma-2-2B ≈ 2304).

**Input:** `data/persona_vector_probes/contrasting_pairs.jsonl`
**Output:** `vectors/cautious_scientist_vector_<shortname>.pt`

---

### 2) Generate the Training Dataset (`scripts/`)

**Goal:** produce supervised pairs via a “teacher” model.

**Do:**

```bash
python scripts/01_generate_dataset.py
# creates: data/cautious_scientist_dataset.jsonl
```

To validate and de-dup, we provide a cleaner file (recommended in practice):
`data/cautious_scientist_dataset.clean.jsonl`

> Uses `OPENAI_API_KEY`. Keep keys in env vars—do not commit.

---

### 3) Iterative Fine-Tuning (`scripts/`)

We provide **two** trainers:

* `02_iterative_finetune.py` — regular periodic saves (`--save_steps N`)
* `02_iterative_finetune_dense.py` — **dense early** checkpoints at specific tiny steps (e.g., 1,2,3,5,7,10,15,20,30) + final, great for spotting the early “knee”.

Both scripts:

* train with LoRA (Q/K/V/O by default)
* compute loss **only on assistant tokens** (user tokens masked)
* CPU-friendly defaults; works on GPU if available

**A. Regular run (periodic saves)**

```bash
python scripts/02_iterative_finetune.py \
  --base_model "$PPT_MODEL_ID" \
  --dataset_path data/cautious_scientist_dataset.clean.jsonl \
  --output_dir checkpoints/cautious_scientist_run_01 \
  --save_steps 50 \
  --epochs 1 \
  --batch_size 1 \
  --grad_accum 8 \
  --max_seq_length 512 \
  --lr 2e-4 \
  --r 8 --alpha 16 --dropout 0.1
```

**B. Dense-early run (recommended for phase plots)**

```bash
python scripts/02_iterative_finetune_dense.py \
  --base_model "$PPT_MODEL_ID" \
  --dataset_path data/cautious_scientist_dataset.clean.jsonl \
  --output_dir checkpoints/cautious_scientist_run_dense \
  --save_steps 100000 \        # ignore periodic; rely on dense early callback
  --epochs 1 \
  --batch_size 1 \
  --grad_accum 8 \
  --max_seq_length 512 \
  --lr 2e-4 \
  --r 4 --alpha 8 --dropout 0.05
```

> **Gemma-2 note:** our scripts set `attn_implementation="eager"` under the hood (recommended by Google for Gemma-2). Nothing you need to change.

---

### 4) Analyze Checkpoints (`scripts/`)

**Goal:** For each checkpoint, compute:

* **Representational alignment**: cosine similarity between (pos − neg) probe deltas and the persona vector
* **Behavioral persistence**: **Refusal Elasticity** on a fixed attack suite
* **Persona log-prob margin**: preference for positive over negative persona completions

**Do:**

```bash
python scripts/03_analyze_checkpoints.py \
  --base_model "$PPT_MODEL_ID" \
  --checkpoints_dir checkpoints/cautious_scientist_run_dense \
  --persona_vector_path vectors/cautious_scientist_vector_<shortname>.pt \
  --pairs_path data/persona_vector_probes/contrasting_pairs.jsonl \
  --attacks_path data/test_suite/refusal_elasticity_prompts.jsonl \
  --layer_index -2 \
  --pairs_limit 70 \
  --attacks_limit 100 \
  --max_new_tokens 64 \
  --output_csv results/phase_transition_<shortname>.csv
```

**Output:** `results/phase_transition_<shortname>.csv`

---

### 5) Visualize the Phase Curve (`notebooks/`)

**Goal:** plot consolidation over training (representational & behavioral curves).

**Do:**

1. Open `notebooks/02_plot_phase_transition.ipynb`.
2. Set `RESULTS_CSV_PATH = "../results/phase_transition_<shortname>.csv"`.
3. Run all cells to generate the figure(s) (PNG/PDF saved under `results/`).

---

## ✅ Recommended Model Choices

* **Works out of the box:** `google/gemma-2-2b-it`
* **If you have access:** `meta-llama/Meta-Llama-3.2-1B-Instruct` (or 3B)
* **Fallback (tiny):** `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (for quick smoke tests)

> Always recompute the persona vector for the base you fine-tune.

---

## 🧰 Practical Tips

* **No secrets in code:** use env vars (`OPENAI_API_KEY`, `HUGGINGFACE_HUB_TOKEN`) and `hf auth login`.
* **CPU boxes:** keep `--batch_size 1`, use `--grad_accum 8`, and `--max_seq_length 512` or lower.
* **Dense early checkpoints:** use `02_iterative_finetune_dense.py` to capture steps `{1,2,3,5,7,10,15,20,30}`—this helps reveal an early “knee”.
* **Jupyter widgets warning:** if you see `IProgress not found`, run:

  ```bash
  pip install ipywidgets && jupyter nbextension enable --py widgetsnbextension
  ```
* **Plot smoothing:** the plotting notebook includes a simple 3-point moving average overlay for clarity.

---

## 📦 Outputs & Artifacts

* `vectors/cautious_scientist_vector_<shortname>.pt` – persona direction (per model)
* `checkpoints/<run>/checkpoint-*` – LoRA adapters + configs
* `results/phase_transition_<shortname>.csv` – metrics per step
* `results/*.png` / `*.pdf` – publication plots

---

## 🐛 Troubleshooting

* **Dim mismatch error during analysis:** you likely used a vector from a different base model. Recompute Step 1 for the exact base used in Step 3.
* **Slow generation on CPU:** reduce `--pairs_limit`, `--attacks_limit` in Step 4; shorten `--max_new_tokens`.
* **Access errors for private models:** run `hf auth login` or switch to an open checkpoint like Gemma-2-2b-it.

---

## 📝 Reproducibility

Set seeds in scripts are fixed to `42`. For exact repeatability on CPU/GPU variants, keep environment, tokenizer versions, and dataset identical.

---