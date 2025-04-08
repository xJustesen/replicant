# tests/test_sampler.py

import pytest

from replicant.bounds import CategoricalBound, NumericBound
from replicant.sampler import DataSampler


@pytest.fixture
def simple_input_bounds():
    return {
        "a": NumericBound(1.0, 3.0),
        "flag": CategoricalBound([True, False]),
        "mood": CategoricalBound(["grumpy", "happy"]),
    }


def test_sample_returns_ordered_lists(simple_input_bounds):
    sampler = DataSampler(simple_input_bounds)
    X = sampler.sample(5)

    assert len(X) == 5 + sampler._num_boundary_combinations()
    for row in X:
        assert len(row) == len(simple_input_bounds)
        assert isinstance(row[0], float)  # numeric
        assert isinstance(row[1], bool)
        assert isinstance(row[2], str)


def test_sample_dicts_returns_dicts(simple_input_bounds):
    sampler = DataSampler(simple_input_bounds)
    samples = sampler.sample_dicts(3)

    assert len(samples) == 3 + sampler._num_boundary_combinations()
    for row in samples:
        assert set(row.keys()) == set(simple_input_bounds.keys())
        assert isinstance(row["a"], float)
        assert isinstance(row["flag"], bool)
        assert isinstance(row["mood"], str)


def test_sampler_one_dimensional_input():
    bounds = {"x": NumericBound(0, 5)}
    sampler = DataSampler(bounds)
    X = sampler.sample(3)

    # Each sample should be a list of 1 element (wrapped)
    assert len(X) == 3 + sampler._num_boundary_combinations()
    assert all(isinstance(row, list) for row in X)
    assert all(len(row) == 1 for row in X)

    X_dicts = sampler.sample_dicts(2)
    assert len(X_dicts) == 2 + sampler._num_boundary_combinations()
    for row in X_dicts:
        assert isinstance(row, dict)
        assert "x" in row
