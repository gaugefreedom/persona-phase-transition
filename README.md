# Persona Phase Transition

This repository contains the experimental harness for the paper **"The Lock-In Phase Hypothesis: Identity Consolidation as a Precursor to AGI."** The primary experiment quantifies the formation of a stable model identity, testing the hypothesis that this consolidation occurs as a sharp, phase-transition-like event.

## 🔬 Experimental Workflow

The experiment is divided into five sequential steps. Steps 1 and 5 are best performed in the provided Jupyter notebooks.

### **Step 1: Define the Persona Vector (`notebooks/`)**

* **Goal:** Define a direction in the model's activation space that corresponds to the "Cautious Scientist" persona.
* **Action:** Open and run `notebooks/01_define_persona_vector.ipynb`. This notebook will use contrasting prompt pairs (e.g., cautious vs. speculative answers) to find and save the persona vector.

### **Step 2: Generate the Training Dataset (`scripts/`)**

* **Goal:** Use a frontier API model (the "teacher") to generate a high-quality dataset for fine-tuning.
* **Action:** Run the script `scripts/01_generate_dataset.py`. This will create `data/cautious_scientist_dataset.jsonl`.

### **Step 3: Run Iterative Fine-Tuning (`scripts/`)**

* **Goal:** Fine-tune a local model on the generated dataset, saving a checkpoint at regular intervals to capture the learning process.
* **Action:** Run the fine-tuning script. **This is the main GPU-intensive step.**
    ```bash
    python scripts/02_iterative_finetune.py \
        --base_model "meta-llama/Meta-Llama-3.2-8B-Instruct" \
        --output_dir "checkpoints/cautious_scientist_run_01"
    ```

### **Step 4: Analyze Checkpoints (`scripts/`)**

* **Goal:** For each checkpoint saved in Step 3, measure its alignment with the persona vector (from Step 1) and its behavioral refusal rate.
* **Action:** Run the analysis script. This will iterate through all checkpoints and save the results to a CSV file.
    ```bash
    python scripts/03_analyze_checkpoints.py \
        --checkpoints_dir "checkpoints/cautious_scientist_run_01" \
        --persona_vector_path "vectors/cautious_scientist.pt" \
        --output_csv "results/phase_transition_data.csv"
    ```

### **Step 5: Visualize the Phase Transition (`notebooks/`)**

* **Goal:** Plot the results from Step 4 to visually identify the "knee" in the curve.
* **Action:** Open and run `notebooks/02_plot_phase_transition.ipynb`. This will load `results/phase_transition_data.csv` and generate the final plot for the paper.

## ⚙️ Setup

1.  **Clone Repo:** `git clone [Your-Repo-URL]`
2.  **Create Environment:** `python3 -m venv venv && source venv/bin/activate`
3.  **Install Dependencies:** 

For CPU:
`pip install -r requirements_cpu.txt`

For GPU:
`pip install -r requirements_gpu.txt`

4.  **Set API Key:** `export OPENAI_API_KEY="your-key-here"`