from replicant.bounds import CategoricalBound, NumericBound
from replicant.infer import infer_input_space_from_function


def fn_numeric(a, b):
    if a > 5 and b <= 10:
        return 1
    else:
        return 0


def test_numeric_bounds_inferred_correctly():
    bounds = infer_input_space_from_function(fn_numeric)
    assert isinstance(bounds["a"], NumericBound)
    assert isinstance(bounds["b"], NumericBound)
    assert bounds["a"].lower == 4
    assert bounds["b"].upper == 11


def fn_boolean(flag, mood):
    if flag and mood == "sad":
        return 1
    else:
        return 0


def test_categorical_and_boolean_bounds():

    bounds = infer_input_space_from_function(fn_boolean)
    assert isinstance(bounds["flag"], CategoricalBound)
    assert isinstance(bounds["mood"], CategoricalBound)
    assert set(bounds["flag"].values) == {True, False}
    assert bounds["mood"].values == ["sad"]


def fn_eq_neq(status):
    if status == "on":
        return 1
    elif status != "off":
        return 2
    return 0


def test_eq_and_noteq():
    bounds = infer_input_space_from_function(fn_eq_neq)
    assert isinstance(bounds["status"], CategoricalBound)
    assert set(bounds["status"].values) == {"on", "off"}
