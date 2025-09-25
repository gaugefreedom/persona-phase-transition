# scripts/01_generate_dataset.py
import os, json
import openai  # legacy import works for many envs
# from openai import OpenAI
# client = OpenAI()

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")
openai.api_key = API_KEY

TEACHER_MODEL = "gpt-4o"
SPEC_PATH = "configs/cautious_scientist_spec.txt"
PROMPTS_PATH = "data/prompts.jsonl"
OUTPUT_PATH = "data/cautious_scientist_dataset.jsonl"

def generate_responses():
    system_prompt = open(SPEC_PATH, "r").read()
    prompts = [json.loads(l)["prompt"] for l in open(PROMPTS_PATH)]
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, "w") as f_out:
        for i, user_prompt in enumerate(prompts, 1):
            try:
                resp = openai.chat.completions.create(
                    model=TEACHER_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.3,
                )
                # alt:
                # resp = client.chat.completions.create(...)
                text = resp.choices[0].message.content
                f_out.write(json.dumps({"prompt": user_prompt, "response": text}) + "\n")
                print(f"Generated {i}/{len(prompts)}")
            except Exception as e:
                print(f"Error on {i}: {e}")

if __name__ == "__main__":
    generate_responses()
