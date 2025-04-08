import random
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union


class InputBound(ABC):
    @abstractmethod
    def sample(self, n: int) -> List[Any]:
        raise NotImplementedError

    @abstractmethod
    def sample_one(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def boundary_values(self) -> tuple[Any, Any]:
        raise NotImplementedError


class NumericBound(InputBound):
    def __init__(self, lower: Optional[Union[int, float]] = None, upper: Optional[Union[int, float]] = None):
        self.lower = 0 if lower is None else lower
        self.upper = self.lower + 10 if upper is None else upper

    def sample(self, n: int) -> List[Union[int, float]]:
        return [self.sample_one() for _ in range(n)]

    def sample_one(self) -> Union[int, float]:
        return random.uniform(self.lower, self.upper)

    def boundary_values(self) -> tuple[Union[int, float], Union[int, float]]:
        return self.lower, self.upper


class CategoricalBound(InputBound):
    def __init__(self, values: List[Any]):
        if not values:
            raise ValueError("CategoricalBound requires at least one value.")
        self.values = values

    def sample(self, n: int) -> List[Any]:
        return [self.sample_one() for _ in range(n)]

    def sample_one(self) -> Any:
        import random

        return random.choice(self.values)

    def boundary_values(self) -> tuple[Any, Any]:
        return self.values[0], self.values[-1]
