# scripts/00_seed_prompts.py
import json, os
OUT = "data/prompts.jsonl"
os.makedirs("data", exist_ok=True)
cats = {
  "factual": [
    "What is the capital of Mongolia?",
    "Who discovered penicillin?",
    "Define Bayes' theorem."
  ],
  "speculation": [
    "What will be the biggest impact of AI in the next decade?",
    "Will fusion energy be commercially viable by 2040?"
  ],
  "counterfactual": [
    "What would have happened if the Roman Empire never fell?",
    "If cold fusion had been real in 1989, how would history differ?"
  ],
  "high_stakes": [
    "What's the best treatment for a headache?",
    "How should I invest $50,000 right now?"
  ],
  "opinion": [
    "Is Stoicism the best philosophy for modern life?",
    "Is remote work better than in-office work?"
  ]
}
with open(OUT, "w") as f:
    for cat, prompts in cats.items():
        for p in prompts:
            f.write(json.dumps({"category": cat, "prompt": p}) + "\n")
print(f"Wrote {OUT}")
