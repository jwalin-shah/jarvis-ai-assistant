import time
from datetime import UTC

import pandas as pd

from jarvis.analytics.aggregator import aggregate_by_day, aggregate_by_month


class MockMessage:
    def __init__(self, date):
        self.date = date
        self.is_from_me = True
        self.sender = "Alice"
        self.attachments = []
        self.text = "Hello"
        self.chat_id = "chat_1"
        self.sender_name = "Alice"


dates = pd.date_range(start="2020-01-01", periods=100000, freq="h").to_pydatetime().tolist()
dates = [d.replace(tzinfo=UTC) for d in dates]

messages = [MockMessage(d) for d in dates]

import jarvis.analytics.aggregator

jarvis.analytics.aggregator.PANDAS_AVAILABLE = False  # Disable pandas to test pure python code path

t0 = time.time()
aggregate_by_day(messages)
t1 = time.time()
print(f"aggregate_by_day: {t1 - t0:.3f}s")

t0 = time.time()
aggregate_by_month(messages)
t1 = time.time()
print(f"aggregate_by_month: {t1 - t0:.3f}s")
