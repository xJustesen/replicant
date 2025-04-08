from replicant.decorator import mimic


@mimic(task="classifier")
def square_classifier(x: float) -> str:
    if x > 5:
        return "high"
    else:
        return "low"


@mimic(task="regressor", mode="blackbox", input_space={"x": (-5, 5)}, n_samples=100_000)
def square_regressor(x: float) -> float:
    return x**2


def test_replicant_classifier():
    assert square_classifier(2) == "low"
    assert square_classifier(7) == "high"


def test_replicant_regressor():
    result = square_regressor(4)
    assert isinstance(result, (float, int))
    assert abs(result - 16) < 1e-2
