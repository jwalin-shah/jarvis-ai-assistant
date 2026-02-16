from __future__ import annotations

from api.services.tracing import TraceStore


def test_trace_store_isolates_concurrent_trace_ids() -> None:
    store = TraceStore(max_traces=10)

    store.start_trace("trace-a", "/a")
    store.start_trace("trace-b", "/b")

    step_a = store.add_step("step-a", trace_id="trace-a", input_summary="in-a")
    step_b = store.add_step("step-b", trace_id="trace-b", input_summary="in-b")

    store.end_step(step_a, output_summary="out-a")
    store.end_step(step_b, output_summary="out-b")

    store.end_trace("trace-a", success=True)
    store.end_trace("trace-b", success=False, error="boom")

    traces = store.get_traces(limit=10)
    trace_map = {trace["trace_id"]: trace for trace in traces}

    assert set(trace_map) == {"trace-a", "trace-b"}
    assert trace_map["trace-a"]["steps"][0]["name"] == "step-a"
    assert trace_map["trace-b"]["steps"][0]["name"] == "step-b"
    assert trace_map["trace-b"]["success"] is False
    assert trace_map["trace-b"]["error"] == "boom"
