from functools import wraps
from typing import Any, Callable, Dict, Literal, Optional

from replicant.bounds import CategoricalBound, InputBound, NumericBound
from replicant.infer import infer_input_space_from_function
from replicant.sampler import DataSampler
from replicant.trainer import fit_function_to_model
from replicant.wrapper import MLFunctionWrapper


def input_space_to_bounds(input_space: Dict[str, Any]) -> Dict[str, InputBound]:
    bounds = {}
    for key, val in input_space.items():
        if isinstance(val, InputBound):
            bounds[key] = val
        elif isinstance(val, tuple) and len(val) == 2 and all(isinstance(v, (int, float)) for v in val):
            bounds[key] = NumericBound(val[0], val[1])
        elif isinstance(val, list):
            bounds[key] = CategoricalBound(val)
        else:
            raise ValueError(f"Cannot convert input space value for '{key}': {val}")
    return bounds


def mimic(
    *,
    input_space: Optional[Dict[str, Any]] = None,
    model_class: Optional[type] = None,
    model_kwargs: Optional[dict] = None,
    task: Literal["classifier", "regressor", "auto"] = "auto",
    mode: Literal["ast", "blackbox"] = "ast",  # future mode: "blackbox", etc.
    n_samples: int = 1000,
) -> Callable:
    """
    Decorator to turn any function into a machine-learned version of itself.

    Parameters:
    - input_space: Manually specified input bounds (overrides inference)
    - model_class: Optional custom model class
    - model_kwargs: Args for the model class
    - task_type: 'classifier', 'regressor', or 'auto'
    - mode: Currently only 'ast' is supported

    Returns:
    - Callable object that wraps the ML model
    """

    def decorator(fn: Callable):
        @wraps(fn)
        def wrapped_fn(*args, **kwargs):
            raise RuntimeError(
                "The wrapped function cannot be called directly. Apply the decorator at module level."
            )

        # Infer or use provided input space
        if mode == "blackbox":
            if input_space is None:
                raise ValueError("input_space must be provided when using mode='blackbox'")
            else:
                bounds = input_space_to_bounds(input_space)
        elif mode == "ast":
            bounds = (
                input_space_to_bounds(input_space) if input_space else infer_input_space_from_function(fn)
            )
        else:
            raise ValueError(f"unsupported mode: {mode!r}")

        # Sample synthetic data
        sampler = DataSampler(bounds)
        X = sampler.sample(n_samples)

        # Fit the function to a model
        model, feature_names = fit_function_to_model(
            fn,
            X,
            sampler.feature_names,
            model_class=model_class,
            model_kwargs=model_kwargs,
            task_type=task,
        )

        # Return the wrapped callable
        return MLFunctionWrapper(model, feature_names)

    return decorator
