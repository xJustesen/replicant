from typing import Any, Callable, Iterable, Literal, Optional, Type

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


class MLModelProtocol:
    def fit(self, X, y) -> Any: ...
    def predict(self, X) -> list[Any]: ...


def fit_function_to_model(
    fn: Callable,
    X: Iterable[Iterable[Any]],
    feature_names: list[str],
    model_class: Optional[Type[MLModelProtocol]] = None,
    model_kwargs: Optional[dict] = None,
    task_type: Literal["classifier", "regressor", "auto"] = "auto",  # "classifier", "regressor", or "auto"
) -> tuple[MLModelProtocol, list[str]]:
    """
    Fit any function using an ML model trained on synthetic data.

    Parameters:
    - fn: A Python function taking named args, e.g., def f(x, y): ...
    - X: Synthetic feature matrix (list of rows)
    - feature_names: List of names for each feature (columns)
    - model_class: Optional model backend (must support fit/predict)
    - model_kwargs: Optional kwargs for model constructor
    - task_type: 'classifier', 'regressor', or 'auto'

    Returns:
    - (trained model instance, feature_names)
    """
    model_kwargs = model_kwargs or {}

    # Generate y by evaluating fn(**kwargs) on each input row
    y = [fn(**dict(zip(feature_names, row))) for row in X]

    # Guess classifier vs regressor if needed
    if task_type == "auto":
        task_type = "classifier" if all(isinstance(val, (int, bool, str)) for val in y) else "regressor"

    if model_class is None:
        model_class = {"classifier": DecisionTreeClassifier, "regressor": DecisionTreeRegressor}[task_type]

    model = model_class(**model_kwargs)  # type: ignore
    model.fit(X, y)

    return model, feature_names
