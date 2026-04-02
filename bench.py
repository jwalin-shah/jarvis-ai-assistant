import timeit

setup = """
text = "Loved that message you sent"
patterns_list = ["Laughed at", "Loved", "Liked", "Disliked", "Emphasized", "Questioned"]
patterns_tuple = tuple(patterns_list)
"""

code1 = "any(text.startswith(p) for p in patterns_list)"
code2 = "text.startswith(patterns_tuple)"

t1 = timeit.timeit(code1, setup=setup, number=1000000)
t2 = timeit.timeit(code2, setup=setup, number=1000000)

print(f"Generator + any(): {t1:.4f}s")
print(f"Tuple + startswith(): {t2:.4f}s")
print(f"Speedup: {t1/t2:.2f}x")
