"""Run a conversation through a local MLX model for fact extraction."""

import mlx_lm

MODEL = "mlx-community/LFM2-350M-4bit"

CONVERSATION = """[2025-07-20 18:49] Jwalin: Ur just like the person I talked to ages ago whose from LA
[2025-07-20 18:50] Radhika: all I remember is the dance you did in runis sweet 16 with FALTU
[2025-07-20 18:50] Jwalin: At some random balbat party
[2025-07-20 18:50] Jwalin: Whomst is faltu
[2025-07-20 18:50] Radhika: THE DANCE THE SONG
[2025-07-20 18:51] Radhika: What's your Instagram
[2025-07-20 18:52] Jwalin: Jwalin.shah
[2025-07-20 18:53] Radhika: also how do u know Shilpan shah? He's my cousin 🤣
[2025-07-20 18:54] Jwalin: I barely know him I just met him once at my cousins place
[2025-07-20 18:54] Jwalin: For his wife's bday"""

PROMPT = f"""Read this chat and list facts about each person. Only list things directly stated. Do not make anything up.

Chat:
{CONVERSATION}

Facts about Radhika:
-"""

print(f"Loading model: {MODEL}", flush=True)
model, tokenizer = mlx_lm.load(MODEL)

print("Generating...\n", flush=True)
response = mlx_lm.generate(
    model,
    tokenizer,
    prompt=PROMPT,
    max_tokens=512,
    verbose=False,
)
print(response)
