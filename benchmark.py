import timeit

setup = """
REACTION_PATTERNS_LIST = ["Laughed at", "Loved", "Liked", "Disliked", "Emphasized", "Questioned"]
REACTION_PATTERNS_TUPLE = ("Laughed at", "Loved", "Liked", "Disliked", "Emphasized", "Questioned")
text = "Laughed at this message"
"""

test_any = "any(text.startswith(p) for p in REACTION_PATTERNS_LIST)"
test_tuple = "text.startswith(REACTION_PATTERNS_TUPLE)"

print("any():", timeit.timeit(test_any, setup=setup, number=1000000))
print("tuple:", timeit.timeit(test_tuple, setup=setup, number=1000000))
