from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from replicant.trainer import fit_function_to_model


def test_classifier_mode_explicit():
    def logic_fn(x):
        return "yes" if x > 5 else "no"

    X = [[3], [7], [4], [8]]
    model, names = fit_function_to_model(logic_fn, X, ["x"], task_type="classifier")
    assert isinstance(model, DecisionTreeClassifier)
    assert names == ["x"]
    assert list(model.predict([[2], [10]])) == ["no", "yes"]


def test_regressor_mode_explicit():
    def math_fn(x):
        return x**2

    X = [[1], [2], [3]]
    model, names = fit_function_to_model(math_fn, X, ["x"], task_type="regressor")
    assert isinstance(model, DecisionTreeRegressor)
    assert names == ["x"]
    assert all(isinstance(y, float) or isinstance(y, int) for y in model.predict([[4]]))


def test_auto_classifier_detection():
    def bool_logic(x):
        return x > 5

    X = [[2], [6], [3], [10]]
    model, _ = fit_function_to_model(bool_logic, X, ["x"])
    assert isinstance(model, DecisionTreeClassifier)
    assert list(model.predict([[1], [7]])) == [False, True]


def test_custom_model_backend():
    class MockModel:
        def __init__(self):
            self.was_fit = False

        def fit(self, X, y):
            self.was_fit = True
            self.X = X
            self.y = y

        def predict(self, X):
            return ["mock"] * len(X)

    def fn(x):
        return "data"

    X = [[1], [2]]
    model, _ = fit_function_to_model(fn, X, ["x"], model_class=MockModel)
    assert isinstance(model, MockModel)
    assert model.was_fit
    assert model.predict([[1], [2]]) == ["mock", "mock"]
