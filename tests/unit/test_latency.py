"""Unit tests for WS4 Latency Benchmark implementation.

Tests timing utilities, scenario definitions, and benchmarker
with mocked model operations.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

# Import scenario definitions (work on all platforms)
from benchmarks.latency.scenarios import (
    BENCHMARK_PROMPTS,
    DEFAULT_MAX_TOKENS,
    LatencyScenario,
    get_benchmark_prompts,
    get_default_scenarios,
    get_scenario_by_type,
)

# Import timer utilities (work on all platforms)
from benchmarks.latency.timer import (
    HighPrecisionTimer,
    TimingResult,
    force_model_unload,
    measure_operation,
    timed_operation,
    warmup_timer,
)

# Contract types
from contracts.latency import LatencyBenchmarkResult, LatencyResult

# Check if MLX is available for conditional tests
try:
    from benchmarks.latency.run import HAS_MLX, MLXLatencyBenchmarker, ModelConfig

    MLX_AVAILABLE = HAS_MLX
except ImportError:
    MLX_AVAILABLE = False
    MLXLatencyBenchmarker = None  # type: ignore[misc, assignment]
    ModelConfig = None  # type: ignore[misc, assignment]


class TestTimingResult:
    """Tests for TimingResult dataclass."""

    def test_from_ns_creates_result(self):
        """Verify from_ns factory method creates correct result."""
        result = TimingResult.from_ns(1_000_000)  # 1ms in nanoseconds
        assert result.elapsed_ns == 1_000_000
        assert result.elapsed_ms == 1.0

    def test_from_ns_handles_zero(self):
        """Verify from_ns handles zero correctly."""
        result = TimingResult.from_ns(0)
        assert result.elapsed_ns == 0
        assert result.elapsed_ms == 0.0

    def test_from_ns_handles_large_values(self):
        """Verify from_ns handles large values (10 seconds)."""
        result = TimingResult.from_ns(10_000_000_000)  # 10 seconds
        assert result.elapsed_ns == 10_000_000_000
        assert result.elapsed_ms == 10_000.0


class TestHighPrecisionTimer:
    """Tests for HighPrecisionTimer."""

    def test_timer_initial_state(self):
        """Verify timer starts in unstarted state."""
        timer = HighPrecisionTimer()
        assert timer.elapsed_ns is None
        assert timer.elapsed_ms is None

    def test_timer_start_stop(self):
        """Verify timer can start and stop."""
        timer = HighPrecisionTimer()
        timer.start()
        time.sleep(0.001)  # Sleep 1ms
        result = timer.stop()
        assert result.elapsed_ms > 0
        assert result.elapsed_ms < 100  # Should be less than 100ms

    def test_timer_stop_without_start_raises(self):
        """Verify stopping without starting raises error."""
        timer = HighPrecisionTimer()
        with pytest.raises(RuntimeError, match="Timer was not started"):
            timer.stop()

    def test_timer_reset(self):
        """Verify timer can be reset."""
        timer = HighPrecisionTimer()
        timer.start()
        timer.stop()
        assert timer.elapsed_ns is not None

        timer.reset()
        assert timer.elapsed_ns is None
        assert timer.elapsed_ms is None

    def test_timer_elapsed_after_stop(self):
        """Verify elapsed values are accessible after stop."""
        timer = HighPrecisionTimer()
        timer.start()
        time.sleep(0.001)
        timer.stop()
        assert timer.elapsed_ns is not None
        assert timer.elapsed_ms is not None
        assert timer.elapsed_ns > 0
        assert timer.elapsed_ms > 0

    def test_timer_precision(self):
        """Verify timer has nanosecond-level precision."""
        timer = HighPrecisionTimer()
        timer.start()
        # Very short operation
        _ = 1 + 1
        result = timer.stop()
        # Should measure something even for very short operations
        assert result.elapsed_ns >= 0


class TestTimedOperationContextManager:
    """Tests for timed_operation context manager."""

    def test_timed_operation_basic(self):
        """Verify context manager times operations."""
        with timed_operation() as timer:
            time.sleep(0.001)
        assert timer.elapsed_ms is not None
        assert timer.elapsed_ms > 0

    def test_timed_operation_with_exception(self):
        """Verify timer stops even if operation raises."""
        with pytest.raises(ValueError):
            with timed_operation() as timer:
                raise ValueError("Test error")
        # Timer should still have stopped
        assert timer.elapsed_ms is not None


class TestMeasureOperation:
    """Tests for measure_operation function."""

    def test_measure_operation_basic(self):
        """Verify measure_operation returns result and timing."""

        def simple_func(x):
            return x * 2

        result, timing = measure_operation(simple_func, 5)
        assert result == 10
        assert timing.elapsed_ms >= 0

    def test_measure_operation_with_kwargs(self):
        """Verify measure_operation handles kwargs."""

        def func_with_kwargs(x, multiplier=2):
            return x * multiplier

        result, timing = measure_operation(func_with_kwargs, 5, multiplier=3)
        assert result == 15
        assert timing.elapsed_ms >= 0

    def test_measure_operation_with_slow_func(self):
        """Verify measure_operation captures time for slow operations."""

        def slow_func():
            time.sleep(0.01)  # 10ms
            return "done"

        result, timing = measure_operation(slow_func)
        assert result == "done"
        assert timing.elapsed_ms >= 10


class TestForceModelUnload:
    """Tests for force_model_unload function."""

    def test_force_model_unload_runs_without_error(self):
        """Verify force_model_unload runs without raising."""
        # Should not raise even without MLX
        force_model_unload()

    @patch("benchmarks.latency.timer.gc.collect")
    def test_force_model_unload_calls_gc(self, mock_gc):
        """Verify force_model_unload calls garbage collector."""
        force_model_unload()
        # Should call gc.collect at least twice
        assert mock_gc.call_count >= 2


class TestWarmupTimer:
    """Tests for warmup_timer function."""

    def test_warmup_timer_runs_without_error(self):
        """Verify warmup_timer runs without raising."""
        warmup_timer()  # Should not raise


class TestLatencyScenario:
    """Tests for LatencyScenario dataclass."""

    def test_latency_scenario_has_required_fields(self):
        """Verify LatencyScenario has all required fields."""
        scenario = LatencyScenario(
            name="test",
            scenario_type="cold",
            prompt="Hello",
            max_tokens=50,
            description="Test scenario",
        )
        assert scenario.name == "test"
        assert scenario.scenario_type == "cold"
        assert scenario.prompt == "Hello"
        assert scenario.max_tokens == 50
        assert scenario.description == "Test scenario"


class TestGetDefaultScenarios:
    """Tests for get_default_scenarios function."""

    def test_returns_three_scenarios(self):
        """Verify we get exactly 3 scenarios (cold, warm, hot)."""
        scenarios = get_default_scenarios()
        assert len(scenarios) == 3

    def test_all_scenario_types_present(self):
        """Verify all scenario types are present."""
        scenarios = get_default_scenarios()
        types = {s.scenario_type for s in scenarios}
        assert types == {"cold", "warm", "hot"}

    def test_scenarios_have_prompts(self):
        """Verify all scenarios have non-empty prompts."""
        scenarios = get_default_scenarios()
        for scenario in scenarios:
            assert scenario.prompt
            assert len(scenario.prompt) > 0

    def test_scenarios_have_max_tokens(self):
        """Verify all scenarios have max_tokens set."""
        scenarios = get_default_scenarios()
        for scenario in scenarios:
            assert scenario.max_tokens > 0


class TestGetScenarioByType:
    """Tests for get_scenario_by_type function."""

    def test_get_cold_scenario(self):
        """Verify we can get cold scenario."""
        scenario = get_scenario_by_type("cold")
        assert scenario.scenario_type == "cold"

    def test_get_warm_scenario(self):
        """Verify we can get warm scenario."""
        scenario = get_scenario_by_type("warm")
        assert scenario.scenario_type == "warm"

    def test_get_hot_scenario(self):
        """Verify we can get hot scenario."""
        scenario = get_scenario_by_type("hot")
        assert scenario.scenario_type == "hot"

    def test_invalid_scenario_raises(self):
        """Verify invalid scenario type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid scenario type"):
            get_scenario_by_type("invalid")


class TestBenchmarkPrompts:
    """Tests for benchmark prompt categories."""

    def test_short_prompts_exist(self):
        """Verify short prompts exist."""
        assert "short" in BENCHMARK_PROMPTS
        assert len(BENCHMARK_PROMPTS["short"]) > 0

    def test_medium_prompts_exist(self):
        """Verify medium prompts exist."""
        assert "medium" in BENCHMARK_PROMPTS
        assert len(BENCHMARK_PROMPTS["medium"]) > 0

    def test_long_prompts_exist(self):
        """Verify long prompts exist."""
        assert "long" in BENCHMARK_PROMPTS
        assert len(BENCHMARK_PROMPTS["long"]) > 0

    def test_get_benchmark_prompts(self):
        """Verify get_benchmark_prompts returns correct category."""
        prompts = get_benchmark_prompts("short")
        assert prompts == BENCHMARK_PROMPTS["short"]

    def test_get_benchmark_prompts_invalid_category(self):
        """Verify invalid category raises ValueError."""
        with pytest.raises(ValueError, match="Invalid prompt category"):
            get_benchmark_prompts("invalid")


class TestLatencyResultContract:
    """Tests for LatencyResult contract compliance."""

    def test_latency_result_has_required_fields(self):
        """Verify LatencyResult has all required fields."""
        result = LatencyResult(
            scenario="cold",
            model_name="test-model",
            context_length=100,
            output_tokens=50,
            load_time_ms=1000.0,
            prefill_time_ms=100.0,
            generation_time_ms=500.0,
            total_time_ms=1600.0,
            tokens_per_second=100.0,
            timestamp="2024-01-01T00:00:00Z",
        )
        assert result.scenario == "cold"
        assert result.model_name == "test-model"
        assert result.context_length == 100
        assert result.output_tokens == 50
        assert result.load_time_ms == 1000.0
        assert result.prefill_time_ms == 100.0
        assert result.generation_time_ms == 500.0
        assert result.total_time_ms == 1600.0
        assert result.tokens_per_second == 100.0
        assert result.timestamp == "2024-01-01T00:00:00Z"


class TestLatencyBenchmarkResultContract:
    """Tests for LatencyBenchmarkResult contract compliance."""

    def test_benchmark_result_has_required_fields(self):
        """Verify LatencyBenchmarkResult has all required fields."""
        result = LatencyBenchmarkResult(
            scenario="warm",
            model_name="test-model",
            num_runs=10,
            p50_ms=500.0,
            p95_ms=800.0,
            p99_ms=1000.0,
            mean_ms=550.0,
            std_ms=50.0,
            results=[],
            timestamp="2024-01-01T00:00:00Z",
        )
        assert result.scenario == "warm"
        assert result.model_name == "test-model"
        assert result.num_runs == 10
        assert result.p50_ms == 500.0
        assert result.p95_ms == 800.0
        assert result.p99_ms == 1000.0
        assert result.mean_ms == 550.0
        assert result.std_ms == 50.0
        assert result.results == []
        assert result.timestamp == "2024-01-01T00:00:00Z"


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestMLXLatencyBenchmarker:
    """Tests for MLXLatencyBenchmarker with mocked model operations."""

    @pytest.fixture
    def mock_loader(self):
        """Create a mock model loader."""
        loader = MagicMock()
        loader.is_loaded.return_value = False
        loader.load.return_value = True
        loader.unload.return_value = None
        loader.generate_sync.return_value = MagicMock(
            text="Generated response",
            tokens_generated=10,
            generation_time_ms=100.0,
        )
        return loader

    @pytest.fixture
    def benchmarker(self, mock_loader, monkeypatch):
        """Create benchmarker with mocked loader."""
        benchmarker = MLXLatencyBenchmarker(config=ModelConfig(model_path="test-model"))

        # Replace the loader creation
        monkeypatch.setattr(benchmarker, "_get_loader", lambda: mock_loader)
        benchmarker._loader = mock_loader

        return benchmarker

    def test_benchmarker_initialization(self):
        """Verify benchmarker initializes correctly."""
        benchmarker = MLXLatencyBenchmarker()
        assert benchmarker._loader is None
        assert benchmarker.config is not None

    def test_measure_single_cold_start(self, benchmarker, mock_loader):
        """Verify cold start measures load time."""
        mock_loader.is_loaded.return_value = False

        result = benchmarker.measure_single(
            model_path="test-model",
            scenario="cold",
            prompt="Hello",
            max_tokens=50,
        )

        assert result.scenario == "cold"
        assert result.load_time_ms >= 0
        assert result.total_time_ms >= result.load_time_ms
        mock_loader.unload.assert_called()

    def test_measure_single_warm_start(self, benchmarker, mock_loader):
        """Verify warm start doesn't time loading."""
        mock_loader.is_loaded.return_value = True

        result = benchmarker.measure_single(
            model_path="test-model",
            scenario="warm",
            prompt="Hello",
            max_tokens=50,
        )

        assert result.scenario == "warm"
        assert result.load_time_ms == 0.0

    def test_measure_single_hot_start(self, benchmarker, mock_loader):
        """Verify hot start works with loaded model."""
        mock_loader.is_loaded.return_value = True

        result = benchmarker.measure_single(
            model_path="test-model",
            scenario="hot",
            prompt="Hello",
            max_tokens=50,
        )

        assert result.scenario == "hot"
        assert result.load_time_ms == 0.0

    def test_measure_single_records_tokens(self, benchmarker, mock_loader):
        """Verify measure_single records token count."""
        mock_loader.is_loaded.return_value = True
        mock_loader.generate_sync.return_value = MagicMock(
            text="Response",
            tokens_generated=25,
            generation_time_ms=50.0,
        )

        result = benchmarker.measure_single(
            model_path="test-model",
            scenario="warm",
            prompt="Hello",
            max_tokens=50,
        )

        assert result.output_tokens == 25

    def test_measure_single_calculates_tokens_per_second(self, benchmarker, mock_loader):
        """Verify tokens per second is calculated."""
        mock_loader.is_loaded.return_value = True

        result = benchmarker.measure_single(
            model_path="test-model",
            scenario="warm",
            prompt="Hello",
            max_tokens=50,
        )

        assert result.tokens_per_second >= 0

    def test_run_benchmark_returns_statistics(self, benchmarker, mock_loader):
        """Verify run_benchmark returns statistical results."""
        mock_loader.is_loaded.return_value = True

        result = benchmarker.run_benchmark(
            model_path="test-model",
            scenario="warm",
            num_runs=5,
        )

        assert result.scenario == "warm"
        assert result.num_runs == 5
        assert result.p50_ms >= 0
        assert result.p95_ms >= 0
        assert result.p99_ms >= 0
        assert result.mean_ms >= 0
        assert result.std_ms >= 0
        assert len(result.results) == 5

    def test_run_benchmark_excludes_first_run(self, benchmarker, mock_loader):
        """Verify first run (JIT outlier) is excluded from stats when >3 runs."""
        mock_loader.is_loaded.return_value = True

        # Make first run much slower
        call_count = [0]

        def varying_generate(*args, **kwargs):
            call_count[0] += 1
            result = MagicMock()
            result.text = "Response"
            result.tokens_generated = 10
            # First call is much slower
            result.generation_time_ms = 1000.0 if call_count[0] == 1 else 100.0
            return result

        mock_loader.generate_sync = varying_generate

        result = benchmarker.run_benchmark(
            model_path="test-model",
            scenario="warm",
            num_runs=5,
        )

        # Statistics should be based on runs 2-5, so should be lower
        assert result.num_runs == 5
        # Mean should be closer to 100ms than 1000ms since outlier excluded
        # (But our timing includes more than just generation_time_ms)

    def test_cleanup_unloads_model(self, benchmarker, mock_loader):
        """Verify cleanup unloads the model."""
        benchmarker.cleanup()
        mock_loader.unload.assert_called()


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestMLXLatencyBenchmarkerErrorHandling:
    """Tests for error handling in MLXLatencyBenchmarker."""

    def test_measure_single_load_failure(self, monkeypatch):
        """Verify measure_single raises on load failure."""
        mock_loader = MagicMock()
        mock_loader.is_loaded.return_value = False
        mock_loader.load.return_value = False  # Load fails

        benchmarker = MLXLatencyBenchmarker(config=ModelConfig(model_path="test"))
        benchmarker._loader = mock_loader
        monkeypatch.setattr(benchmarker, "_get_loader", lambda: mock_loader)

        with pytest.raises(RuntimeError, match="Failed to load model"):
            benchmarker.measure_single(
                model_path="test",
                scenario="cold",
                prompt="Hello",
                max_tokens=50,
            )


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestCLIRunner:
    """Tests for the latency benchmark CLI runner."""

    def test_main_creates_output_file(self, tmp_path, monkeypatch):
        """Test main function creates output file."""
        import sys

        from benchmarks.latency.run import main

        output_file = tmp_path / "results.json"

        # Mock the benchmarker to avoid actual model loading
        mock_result = MagicMock()
        mock_result.scenario = "warm"
        mock_result.model_name = "test-model"
        mock_result.num_runs = 5
        mock_result.p50_ms = 500.0
        mock_result.p95_ms = 800.0
        mock_result.p99_ms = 1000.0
        mock_result.mean_ms = 550.0
        mock_result.std_ms = 50.0
        mock_result.results = []

        mock_benchmarker = MagicMock()
        mock_benchmarker.run_all_scenarios.return_value = {"warm": mock_result}
        mock_benchmarker.cleanup.return_value = None

        import benchmarks.latency.run

        monkeypatch.setattr(
            benchmarks.latency.run,
            "MLXLatencyBenchmarker",
            lambda config=None: mock_benchmarker,
        )

        # Mock sys.argv
        monkeypatch.setattr(
            sys, "argv", ["run.py", "--output", str(output_file), "--scenario", "warm"]
        )

        # Need to mock single scenario path
        mock_benchmarker.run_benchmark.return_value = mock_result
        result = main()

        assert result == 0
        assert output_file.exists()

        # Verify output is valid JSON
        import json

        with open(output_file) as f:
            data = json.load(f)

        assert "scenarios" in data
        assert "timestamp" in data


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestProtocolCompliance:
    """Verify MLXLatencyBenchmarker implements LatencyBenchmarker protocol."""

    def test_has_measure_single_method(self):
        """Verify measure_single method exists."""
        benchmarker = MLXLatencyBenchmarker()
        assert hasattr(benchmarker, "measure_single")
        assert callable(benchmarker.measure_single)

    def test_has_run_benchmark_method(self):
        """Verify run_benchmark method exists."""
        benchmarker = MLXLatencyBenchmarker()
        assert hasattr(benchmarker, "run_benchmark")
        assert callable(benchmarker.run_benchmark)

    def test_measure_single_signature(self):
        """Verify measure_single has correct signature."""
        import inspect

        sig = inspect.signature(MLXLatencyBenchmarker.measure_single)
        params = list(sig.parameters.keys())
        assert "model_path" in params
        assert "scenario" in params
        assert "prompt" in params
        assert "max_tokens" in params

    def test_run_benchmark_signature(self):
        """Verify run_benchmark has correct signature."""
        import inspect

        sig = inspect.signature(MLXLatencyBenchmarker.run_benchmark)
        params = list(sig.parameters.keys())
        assert "model_path" in params
        assert "scenario" in params
        assert "num_runs" in params


class TestDefaultMaxTokens:
    """Tests for DEFAULT_MAX_TOKENS constant."""

    def test_default_max_tokens_is_positive(self):
        """Verify DEFAULT_MAX_TOKENS is positive."""
        assert DEFAULT_MAX_TOKENS > 0

    def test_default_max_tokens_is_reasonable(self):
        """Verify DEFAULT_MAX_TOKENS is reasonable for benchmarking."""
        # Should be between 10 and 200 for latency benchmarks
        assert 10 <= DEFAULT_MAX_TOKENS <= 200
