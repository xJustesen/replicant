import pytest

from replicant.wrapper import MLFunctionWrapper


class MockModel:
    def predict(self, X):
        return [sum(row) for row in X]  # Return the sum of inputs as prediction


@pytest.fixture
def wrapper():
    model = MockModel()
    return MLFunctionWrapper(model, ["x", "y", "z"])


def test_wrapper_positional(wrapper):
    result = wrapper(1, 2, 3)
    assert result == 6


def test_wrapper_keyword(wrapper):
    result = wrapper(x=1, y=2, z=4)
    assert result == 7


def test_wrapper_mixed(wrapper):
    result = wrapper(1, y=2, z=3)
    assert result == 6


def test_wrapper_missing_argument(wrapper):
    with pytest.raises(ValueError, match="Missing argument: z"):
        wrapper(1, 2)
