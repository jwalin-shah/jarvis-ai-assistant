with open("tests/unit/test_reply_pipeline_flags.py", "r") as f:
    text = f.read()

# Fix the RAG document extraction.
# "similarity" wasn't mapped to "score" in RAGDocument, so we need to pass `{"text": "doc", "score": 0.9}` in the test instead of `{"text": "doc", "similarity": 0.9}`
text = text.replace('{"text": "doc", "similarity": 0.9}', '{"text": "doc", "score": 0.9, "similarity": 0.9}')

with open("tests/unit/test_reply_pipeline_flags.py", "w") as f:
    f.write(text)
