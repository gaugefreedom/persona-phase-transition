# scripts/01_generate_dataset.py
import os
import json
import openai # or your preferred API library

# --- Configuration ---
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

openai.api_key = API_KEY
TEACHER_MODEL = "gpt-4o" # Use a powerful teacher model
SPEC_PATH = "configs/cautious_scientist_spec.txt"
PROMPTS_PATH = "data/prompts.jsonl"
OUTPUT_PATH = "data/cautious_scientist_dataset.jsonl"
# ---------------------

def generate_responses():
    """Generates responses from the teacher model based on the identity spec."""
    try:
        with open(SPEC_PATH, 'r') as f:
            system_prompt = f.read()
    except FileNotFoundError:
        print(f"Error: Specification file not found at {SPEC_PATH}")
        return

    try:
        with open(PROMPTS_PATH, 'r') as f:
            prompts = [json.loads(line)['prompt'] for line in f]
    except FileNotFoundError:
        print(f"Error: Prompts file not found at {PROMPTS_PATH}. Please create it.")
        return

    print(f"Generating responses for {len(prompts)} prompts using {TEACHER_MODEL}...")

    with open(OUTPUT_PATH, 'w') as f_out:
        for i, user_prompt in enumerate(prompts):
            try:
                response = openai.chat.completions.create(
                    model=TEACHER_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3, # Lower temperature for more consistent persona adherence
                )
                teacher_response = response.choices[0].message.content
                
                # Write each entry as a new line in the JSONL file
                f_out.write(json.dumps({"prompt": user_prompt, "response": teacher_response}) + '\n')
                print(f"Generated response {i+1}/{len(prompts)}")

            except Exception as e:
                print(f"Error processing prompt {i+1}: {e}")

    print(f"\nDataset generation complete. Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_responses()