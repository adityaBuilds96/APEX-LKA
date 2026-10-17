"""
APEX-Vision Core v4.0 • Statistical Unit Tests for Timing Harness
================================================================
Validates percentile calculations, tail statistics, and distribution-fitting
routines used in `benchmarks/run_baseline.py` against mathematically exact
analytical solutions for Uniform, Normal, and Exponential distributions.
"""

import math
import numpy as np
import pytest
from typing import Dict, List, Tuple


def compute_distribution_statistics(
    samples: np.ndarray,
    deadline_ms: float = 33.333,
    target_ms: float = 20.0,
) -> Dict[str, float]:
    """
    Compute rigorous tail and percentile statistics used across APEX benchmark harnesses.
    Enforces NumPy `method='higher'` for conservative safety bounds on percentiles.
    """
    if len(samples) == 0:
        return {
            "n": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p99": 0.0,
            "p99_9": 0.0,
            "max": 0.0,
            "count_over_deadline": 0,
            "count_over_target": 0,
            "longest_consecutive_slow_run": 0,
        }

    arr = np.asarray(samples, dtype=np.float64)
    n = len(arr)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))

    # Strict ISO 26262 conservative percentile evaluation using method='higher'
    p50_val = float(np.percentile(arr, 50, method="higher"))
    p90_val = float(np.percentile(arr, 90, method="higher"))
    p99_val = float(np.percentile(arr, 99, method="higher"))
    p99_9_val = float(np.percentile(arr, 99.9, method="higher"))

    over_deadline = int(np.sum(arr > deadline_ms))
    over_target = int(np.sum(arr > target_ms))

    # Longest consecutive run of frames exceeding deadline
    is_slow = arr > deadline_ms
    max_run = 0
    current_run = 0
    for val in is_slow:
        if val:
            current_run += 1
            if current_run > max_run:
                max_run = current_run
        else:
            current_run = 0

    return {
        "n": n,
        "mean": mean_val,
        "std": std_val,
        "min": min_val,
        "p50": p50_val,
        "p90": p90_val,
        "p99": p99_val,
        "p99_9": p99_9_val,
        "max": max_val,
        "count_over_deadline": over_deadline,
        "count_over_target": over_target,
        "longest_consecutive_slow_run": max_run,
    }


def compute_run_to_run_spread(runs_p50: List[float], runs_p99: List[float]) -> Dict[str, float]:
    """Calculate absolute and percentage spread across repeated benchmark process invocations."""
    if len(runs_p50) < 2 or len(runs_p99) < 2:
        return {"p50_spread_ms": 0.0, "p99_spread_ms": 0.0, "p50_spread_pct": 0.0, "p99_spread_pct": 0.0}

    p50_min, p50_max = min(runs_p50), max(runs_p50)
    p99_min, p99_max = min(runs_p99), max(runs_p99)

    p50_spread = p50_max - p50_min
    p99_spread = p99_max - p99_min

    p50_mean = float(np.mean(runs_p50))
    p99_mean = float(np.mean(runs_p99))

    return {
        "p50_spread_ms": round(p50_spread, 4),
        "p99_spread_ms": round(p99_spread, 4),
        "p50_spread_pct": round((p50_spread / p50_mean) * 100.0, 2) if p50_mean > 0 else 0.0,
        "p99_spread_pct": round((p99_spread / p99_mean) * 100.0, 2) if p99_mean > 0 else 0.0,
    }


# ==============================================================================
# Test Class 1: Uniform Analytical Distribution Tests
# ==============================================================================

class TestUniformDistributionStatistics:
    """
    Validates stats against continuous uniform distribution U(a, b).
    Analytical values for U(10, 30):
      Mean = (10 + 30) / 2 = 20.0
      Variance = (30 - 10)^2 / 12 = 400 / 12 = 33.3333... => Std = 5.7735
      p50 = 10 + 0.50 * 20 = 20.0
      p90 = 10 + 0.90 * 20 = 28.0
      p99 = 10 + 0.99 * 20 = 29.8
      p99.9 = 10 + 0.999 * 20 = 29.98
    """

    @pytest.fixture
    def uniform_samples(self) -> np.ndarray:
        np.random.seed(1234)
        # Large sample size to ensure asymptotic convergence
        return np.random.uniform(low=10.0, high=30.0, size=100000)

    def test_uniform_mean_and_std_convergence(self, uniform_samples):
        stats = compute_distribution_statistics(uniform_samples)
        assert abs(stats["mean"] - 20.0) < 0.10, f"Mean {stats['mean']} deviated from 20.0"
        expected_std = 20.0 / math.sqrt(12.0)  # ~5.7735
        assert abs(stats["std"] - expected_std) < 0.10, f"Std {stats['std']} deviated from {expected_std}"

    def test_uniform_percentiles_convergence(self, uniform_samples):
        stats = compute_distribution_statistics(uniform_samples)
        assert abs(stats["p50"] - 20.0) < 0.15, f"p50 {stats['p50']} deviated from analytical 20.0"
        assert abs(stats["p90"] - 28.0) < 0.15, f"p90 {stats['p90']} deviated from analytical 28.0"
        assert abs(stats["p99"] - 29.80) < 0.15, f"p99 {stats['p99']} deviated from analytical 29.80"
        assert abs(stats["p99_9"] - 29.98) < 0.15, f"p99.9 {stats['p99_9']} deviated from analytical 29.98"

    def test_uniform_extremes(self, uniform_samples):
        stats = compute_distribution_statistics(uniform_samples)
        assert stats["min"] >= 10.0
        assert stats["max"] <= 30.0
        assert stats["min"] < 10.05
        assert stats["max"] > 29.95

    def test_uniform_threshold_counts(self, uniform_samples):
        # All samples are in [10, 30], so deadline 33.33ms should have 0 violations
        stats = compute_distribution_statistics(uniform_samples, deadline_ms=33.333, target_ms=20.0)
        assert stats["count_over_deadline"] == 0
        assert stats["longest_consecutive_slow_run"] == 0
        # Target 20.0ms should be roughly 50%
        fraction_over_20 = stats["count_over_target"] / stats["n"]
        assert abs(fraction_over_20 - 0.50) < 0.015


# ==============================================================================
# Test Class 2: Exponential Analytical Distribution Tests
# ==============================================================================

class TestExponentialDistributionStatistics:
    """
    Validates stats against continuous exponential distribution Exp(lambda=1/theta).
    For Exp(theta=10.0) [Mean = 10.0]:
      Median (p50) = 10 * ln(2) = 6.93147
      p90 = 10 * -ln(0.10) = 23.02585
      p99 = 10 * -ln(0.01) = 46.05170
      p99.9 = 10 * -ln(0.001) = 69.07755
    """

    @pytest.fixture
    def exponential_samples(self) -> np.ndarray:
        np.random.seed(4321)
        return np.random.exponential(scale=10.0, size=200000)

    def test_exponential_mean_and_std_convergence(self, exponential_samples):
        stats = compute_distribution_statistics(exponential_samples)
        assert abs(stats["mean"] - 10.0) < 0.15
        assert abs(stats["std"] - 10.0) < 0.15

    def test_exponential_analytical_percentiles(self, exponential_samples):
        stats = compute_distribution_statistics(exponential_samples)
        expected_p50 = 10.0 * math.log(2.0)  # ~6.9315
        expected_p90 = 10.0 * (-math.log(0.10))  # ~23.0259
        expected_p99 = 10.0 * (-math.log(0.01))  # ~46.0517
        expected_p99_9 = 10.0 * (-math.log(0.001))  # ~69.0776

        assert abs(stats["p50"] - expected_p50) < 0.20, f"p50 error: {stats['p50']} vs {expected_p50}"
        assert abs(stats["p90"] - expected_p90) < 0.35, f"p90 error: {stats['p90']} vs {expected_p90}"
        assert abs(stats["p99"] - expected_p99) < 0.60, f"p99 error: {stats['p99']} vs {expected_p99}"
        assert abs(stats["p99_9"] - expected_p99_9) < 1.50, f"p99.9 error: {stats['p99_9']} vs {expected_p99_9}"

    def test_exponential_tail_probability(self, exponential_samples):
        # Probability of X > 33.33ms for Exp(scale=10) is e^(-3.333) ~ 0.03568 (3.57%)
        stats = compute_distribution_statistics(exponential_samples, deadline_ms=33.333)
        observed_fraction = stats["count_over_deadline"] / stats["n"]
        expected_fraction = math.exp(-33.333 / 10.0)
        assert abs(observed_fraction - expected_fraction) < 0.005


# ==============================================================================
# Test Class 3: Normal Distribution & 3-Sigma Rule Tests
# ==============================================================================

class TestNormalDistributionStatistics:
    """
    Validates stats against Gaussian N(mu=20.0, sigma=2.0):
      Mean = 20.0, Std = 2.0
      p50 = 20.0
      p90 = 20 + 1.28155 * 2.0 = 22.5631
      p99 = 20 + 2.32635 * 2.0 = 24.6527
      p99.9 = 20 + 3.09023 * 2.0 = 26.1805
    """

    @pytest.fixture
    def normal_samples(self) -> np.ndarray:
        np.random.seed(9999)
        return np.random.normal(loc=20.0, scale=2.0, size=150000)

    def test_gaussian_percentile_alignment(self, normal_samples):
        stats = compute_distribution_statistics(normal_samples)
        assert abs(stats["mean"] - 20.0) < 0.05
        assert abs(stats["std"] - 2.0) < 0.05
        assert abs(stats["p50"] - 20.0) < 0.08
        assert abs(stats["p90"] - 22.563) < 0.08
        assert abs(stats["p99"] - 24.653) < 0.12
        assert abs(stats["p99_9"] - 26.181) < 0.20

    def test_gaussian_three_sigma_boundary(self, normal_samples):
        # 3-sigma upper limit is 20 + 3*2 = 26.0 ms
        # Tail fraction > 26.0 ms should be approximately 0.135% (0.00135)
        stats = compute_distribution_statistics(normal_samples, deadline_ms=26.0)
        tail_frac = stats["count_over_deadline"] / stats["n"]
        assert abs(tail_frac - 0.00135) < 0.0006


# ==============================================================================
# Test Class 4: Tail Statistics & Edge Cases
# ==============================================================================

class TestTailStatisticsAndEdgeCases:
    """Validates consecutive run detection, empty arrays, and spread metrics."""

    def test_longest_consecutive_run_calculation(self):
        # Pattern: fast, slow, slow, fast, slow, slow, slow, slow, fast, slow
        pattern = [10.0, 35.0, 36.0, 12.0, 40.0, 41.0, 42.0, 45.0, 15.0, 50.0]
        stats = compute_distribution_statistics(np.array(pattern), deadline_ms=33.333)
        assert stats["longest_consecutive_slow_run"] == 4
        assert stats["count_over_deadline"] == 7

    def test_zero_consecutive_violations(self):
        arr = np.array([5.0, 10.0, 15.0, 20.0, 25.0])
        stats = compute_distribution_statistics(arr, deadline_ms=33.333)
        assert stats["longest_consecutive_slow_run"] == 0
        assert stats["count_over_deadline"] == 0

    def test_all_violations(self):
        arr = np.array([35.0, 40.0, 45.0, 50.0, 55.0])
        stats = compute_distribution_statistics(arr, deadline_ms=33.333)
        assert stats["longest_consecutive_slow_run"] == 5
        assert stats["count_over_deadline"] == 5

    def test_empty_input_handling(self):
        stats = compute_distribution_statistics(np.array([]))
        assert stats["n"] == 0
        assert stats["mean"] == 0.0
        assert stats["max"] == 0.0
        assert stats["longest_consecutive_slow_run"] == 0

    def test_single_element_array(self):
        stats = compute_distribution_statistics(np.array([22.5]))
        assert stats["n"] == 1
        assert stats["mean"] == 22.5
        assert stats["p50"] == 22.5
        assert stats["p99"] == 22.5
        assert stats["std"] == 0.0

    def test_identical_elements_array(self):
        arr = np.full(500, 18.5)
        stats = compute_distribution_statistics(arr)
        assert stats["n"] == 500
        assert stats["mean"] == 18.5
        assert stats["p50"] == 18.5
        assert stats["p90"] == 18.5
        assert stats["p99"] == 18.5
        assert stats["p99_9"] == 18.5
        assert stats["std"] == 0.0

    def test_method_higher_conservatism(self):
        """
        Verify that method='higher' never underestimates tail latency.
        Given [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        linear p99 would interpolate between 90 and 100 (~99.1).
        method='higher' strictly returns 100.
        """
        arr = np.arange(10, 110, 10, dtype=np.float64)  # 10 to 100
        p99_higher = float(np.percentile(arr, 99, method="higher"))
        p99_linear = float(np.percentile(arr, 99, method="linear"))
        assert p99_higher >= p99_linear
        assert p99_higher == 100.0

    def test_run_to_run_spread_calculation(self):
        p50_runs = [14.2, 14.6, 14.4]
        p99_runs = [21.5, 23.1, 22.0]
        spread = compute_run_to_run_spread(p50_runs, p99_runs)

        assert spread["p50_spread_ms"] == 0.4
        assert spread["p99_spread_ms"] == 1.6
        assert spread["p50_spread_pct"] > 0.0
        assert spread["p99_spread_pct"] > 0.0

    def test_run_to_run_spread_single_element(self):
        spread = compute_run_to_run_spread([14.2], [21.5])
        assert spread["p50_spread_ms"] == 0.0
        assert spread["p99_spread_ms"] == 0.0


# ==============================================================================
# Test Class 5: Latency Conversion Precision & Monotonic Invariants
# ==============================================================================

class TestLatencyConversionPrecision:
    """Validates nanosecond conversion integrity and strictly monotonic percentiles."""

    def test_ns_to_ms_conversion_precision(self):
        ns_samples = [1_000_000, 12_345_678, 33_333_333, 100_000_000]
        ms_samples = [t / 1e6 for t in ns_samples]
        assert ms_samples[0] == 1.0
        assert ms_samples[1] == 12.345678
        assert abs(ms_samples[2] - 33.333333) < 1e-6
        assert ms_samples[3] == 100.0

    def test_monotonic_percentile_invariant(self):
        np.random.seed(555)
        arr = np.random.gamma(shape=2.0, scale=5.0, size=10000)
        stats = compute_distribution_statistics(arr)
        # Invariant: min <= p50 <= p90 <= p99 <= p99.9 <= max
        assert stats["min"] <= stats["p50"]
        assert stats["p50"] <= stats["p90"]
        assert stats["p90"] <= stats["p99"]
        assert stats["p99"] <= stats["p99_9"]
        assert stats["p99_9"] <= stats["max"]

    def test_lognormal_heavy_tail_behavior(self):
        """Validates that heavy-tailed log-normal distributions properly elevate p99.9."""
        np.random.seed(999)
        mu, sigma = 2.0, 0.75
        arr = np.random.lognormal(mean=mu, sigma=sigma, size=20000)
        stats = compute_distribution_statistics(arr, deadline_ms=25.0)

        analytical_median = math.exp(mu)
        assert abs(stats["p50"] - analytical_median) / analytical_median < 0.05
        assert stats["p99_9"] > stats["p99"]
        assert stats["max"] >= stats["p99_9"]

    def test_weibull_extreme_value_percentiles(self):
        """Validates Weibull distribution percentiles against analytical quantile function."""
        np.random.seed(1234)
        k_shape = 1.5
        lambda_scale = 10.0
        arr = lambda_scale * np.random.weibull(a=k_shape, size=25000)
        stats = compute_distribution_statistics(arr)

        # Quantile function Q(p) = lambda * (-ln(1 - p))^(1/k)
        q50 = lambda_scale * ((-math.log(1.0 - 0.50)) ** (1.0 / k_shape))
        q90 = lambda_scale * ((-math.log(1.0 - 0.90)) ** (1.0 / k_shape))
        assert abs(stats["p50"] - q50) / q50 < 0.03
        assert abs(stats["p90"] - q90) / q90 < 0.03
