"""Algorithm registry for PyFR RL."""

from .registry import (
    AlgorithmSpec,
    create_model,
    get_algorithm_spec,
    is_recurrent_algorithm,
    load_model,
    normalize_algorithm_name,
)

__all__ = [
    'AlgorithmSpec',
    'normalize_algorithm_name',
    'get_algorithm_spec',
    'is_recurrent_algorithm',
    'create_model',
    'load_model',
]
