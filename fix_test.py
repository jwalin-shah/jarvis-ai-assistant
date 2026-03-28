with open("tests/unit/test_reply_pipeline_flags.py", "r") as f:
    text = f.read()

# Fix the search_results similarity float mapping issue:
# In hybrid_search.py or other places, it might be passing dicts. The test does:
# `[{"text": "doc", "similarity": 0.9}]` but the model gets `score = 0.0`.
# Oh, looking at build_generation_request... wait.
