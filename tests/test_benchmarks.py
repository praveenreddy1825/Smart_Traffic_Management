"""
Tests for Module 14: Performance Benchmarking and Metric Calculation Logic.
"""

import numpy as np
import pytest

from scripts.run_benchmarks import compute_stats


def test_compute_stats_accuracy():
    """Verify that compute_stats accurately calculates statistical metrics."""
    data = [10.0, 20.0, 30.0, 40.0, 50.0]
    stats = compute_stats(data)

    assert stats["mean_ms"] == 30.0
    assert stats["min_ms"] == 10.0
    assert stats["max_ms"] == 50.0
    assert stats["p50_ms"] == 30.0
    assert stats["p90_ms"] >= 40.0
    assert stats["p95_ms"] >= 45.0
    assert stats["p99_ms"] <= 50.0


def test_compute_stats_single_value():
    """Verify compute_stats handles single-element arrays without division errors."""
    data = [42.5]
    stats = compute_stats(data)

    assert stats["mean_ms"] == 42.5
    assert stats["min_ms"] == 42.5
    assert stats["max_ms"] == 42.5
    assert stats["std_ms"] == 0.0
    assert stats["p50_ms"] == 42.5

