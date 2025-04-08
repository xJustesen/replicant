from typing import Iterable

import pytest

from replicant.bounds import CategoricalBound, NumericBound


def test_numeric_bound_defaults():
    bound = NumericBound()
    val = bound.sample_one()
    assert 0 <= val <= 10


def test_numeric_bound_explicit():
    bound = NumericBound(5, 7)
    for _ in range(10):
        val = bound.sample_one()
        assert 5 <= val <= 7


def test_numeric_bound_batch():
    bound = NumericBound(1, 2)
    values = bound.sample(5)
    assert isinstance(values, Iterable)
    assert len(values) == 5
    for v in values:
        assert 1 <= v <= 2


def test_categorical_bound_sampling():
    bound = CategoricalBound(["a", "b", "c"])
    for _ in range(10):
        assert bound.sample_one() in ["a", "b", "c"]


def test_categorical_bound_batch():
    bound = CategoricalBound(["yes", "no"])
    samples = bound.sample(5)
    assert isinstance(samples, Iterable)
    assert len(samples) == 5
    for val in samples:
        assert val in ["yes", "no"]


def test_categorical_bound_empty_list():
    with pytest.raises(ValueError):
        CategoricalBound([])
