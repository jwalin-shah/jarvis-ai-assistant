import logging

from jarvis.contacts.fact_storage import get_facts_for_contact
from models.loader import MLXModelLoader, ModelConfig

# Configure logging
logging.basicConfig(level=logging.INFO)

CHAT_ID = "iMessage;-;+15629643639"
MODEL_PATH = "models/lfm2-1.2b-extract-mlx-4bit"

BIO_PROMPT_TEMPLATE = """You are an expert biographer.
Below is a list of verified facts learned about Radhika from her chat history.
Synthesize these facts into a coherent, professional, and insightful Character Profile.

Structure:
1. Current Life Status (Job, Location, Career Stage)
2. Health & Well-being
3. Interests & Personality
4. Key Relationship Context (How she interacts with Jwalin)

Verified Facts about Radhika:
{fact_list}

Character Profile for Radhika:"""


def generate_bio():
    # 1. Fetch Verified Facts from DB
    print(f"Fetching verified facts for {CHAT_ID}...")
    all_facts = get_facts_for_contact(CHAT_ID)

    # Filter only facts about Radhika
    radhika_facts = [f for f in all_facts if f.subject == "Radhika"]

    if not radhika_facts:
        print("No facts found for Radhika in the database.")
        return

    print(f"Found {len(radhika_facts)} facts about Radhika.")

    # Clean and join facts safely
    fact_text_list = []
    for f in radhika_facts:
        clean_val = " ".join(str(f.value).splitlines())
        fact_text_list.append(f"- {clean_val}")

    fact_blob = "\n".join(fact_text_list)

    # 2. Load 1.2B Model
    print("Loading 1.2B model for synthesis...")
    loader = MLXModelLoader(ModelConfig(model_path=MODEL_PATH))
    if not loader.load():
        print("Failed to load model.")
        return

    # 3. Generate Biography
    print("Synthesizing biography...")
    prompt = BIO_PROMPT_TEMPLATE.format(fact_list=fact_blob)

    result = loader.generate_sync(
        prompt=prompt, max_tokens=800, temperature=0.3, stop_sequences=["###", "Verified Facts:"]
    )

    print("\n" + "=" * 60)
    print("RADHIKA SHAH - FINAL CHARACTER BIOGRAPHY")
    print("=" * 60)
    print(result.text.strip())
    print("=" * 60)


if __name__ == "__main__":
    generate_bio()
