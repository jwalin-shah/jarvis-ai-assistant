with open('tests/unit/test_reply_pipeline_flags.py', 'r') as f:
    data = f.read()
data = data.replace('search_results = [{"text": "doc", "similarity": 0.9}]', 'search_results = [{"text": "doc", "similarity": 0.9, "score": 0.9}]')
with open('tests/unit/test_reply_pipeline_flags.py', 'w') as f:
    f.write(data)
