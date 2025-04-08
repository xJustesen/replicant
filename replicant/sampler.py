import itertools
from typing import Any

from replicant.bounds import InputBound


class DataSampler:
    def __init__(self, input_bounds: dict[str, InputBound]):
        self.input_bounds = input_bounds
        self.feature_names = list(input_bounds)

    def sample(self, n: int) -> list[list[Any]]:
        base = self._sample_random(n)
        boundary = self._sample_boundary_combinations()
        return base + boundary

    def _sample_random(self, n: int) -> list[list[Any]]:
        return [[self.input_bounds[name].sample_one() for name in self.feature_names] for _ in range(n)]

    def _sample_boundary_combinations(self) -> list[list[Any]]:
        bounds = [self.input_bounds[name].boundary_values() for name in self.feature_names]
        return [list(comb) for comb in itertools.product(*bounds)]

    def _num_boundary_combinations(self) -> int:
        return 2 ** len(self.feature_names)

    def sample_dicts(self, n: int) -> list[dict[str, Any]]:
        rows = self.sample(n)
        return [{name: val for name, val in zip(self.feature_names, row)} for row in rows]
