import json
import math
import os
from typing import Any

import torch.nn as nn


def model_meta_path(model_path: str) -> str:
    if model_path.endswith('.zip'):
        return f'{model_path[:-4]}.meta.json'
    return f'{model_path}.meta.json'


def resolve_model_path(path: str) -> str:
    if os.path.exists(path):
        return path

    if os.path.exists(f'{path}.zip'):
        return f'{path}.zip'

    return path


def save_metadata(model_path: str, metadata: dict[str, Any]) -> None:
    with open(model_meta_path(model_path), 'w') as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def load_metadata(model_path: str) -> dict[str, Any] | None:
    meta_path = model_meta_path(model_path)
    if not os.path.exists(meta_path):
        return None

    try:
        with open(meta_path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def activation_from_name(name: str):
    if hasattr(nn, name):
        return getattr(nn, name)

    aliases = {
        'relu': nn.ReLU,
        'tanh': nn.Tanh,
        'elu': nn.ELU,
        'gelu': nn.GELU,
        'selu': nn.SELU,
        'silu': nn.SiLU,
        'swish': nn.SiLU,
        'leakyrelu': nn.LeakyReLU,
        'leaky_relu': nn.LeakyReLU,
    }

    key = (name or '').lower()
    if key in aliases:
        return aliases[key]

    print(f"Warning: Unknown activation '{name}'. Falling back to Tanh.")
    return nn.Tanh


def compare_configs(checkpoint_config, current_config):
    """
    Compare two config files line by line and return differences.

    Args:
        checkpoint_config: Config content from checkpoint as string
        current_config: Current config content as string

    Returns:
        List of tuples with (line_number, checkpoint_line, current_line)
        for different lines.
    """
    if not checkpoint_config or not current_config:
        return []

    checkpoint_lines = [line.strip() for line in checkpoint_config.splitlines()]
    current_lines = [line.strip() for line in current_config.splitlines()]

    differences = []

    for i, (ckpt_line, curr_line) in enumerate(zip(checkpoint_lines,
                                                   current_lines)):
        if not ckpt_line or ckpt_line.startswith(';'):
            continue
        if not curr_line or curr_line.startswith(';'):
            continue

        if ckpt_line != curr_line:
            differences.append((i + 1, ckpt_line, curr_line))

    if len(checkpoint_lines) > len(current_lines):
        for i, line in enumerate(checkpoint_lines[len(current_lines):],
                                 start=len(current_lines)):
            if line and not line.startswith(';'):
                differences.append((i + 1, line, '[MISSING]'))
    elif len(current_lines) > len(checkpoint_lines):
        for i, line in enumerate(current_lines[len(checkpoint_lines):],
                                 start=len(checkpoint_lines)):
            if line and not line.startswith(';'):
                differences.append((i + 1, '[MISSING]', line))

    return differences


def get_closest_divisor(n, target):
    """Find the closest divisor of n to target."""
    divisors = set()
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            divisors.add(i)
            divisors.add(n // i)

    return min(divisors, key=lambda x: (abs(x - target), -x))


def get_device_count(backend_name):
    """Get number of available devices for a backend."""
    if backend_name == 'hip':
        from pyfr.backends.hip.driver import HIP
        return HIP().device_count()
    if backend_name == 'cuda':
        from pyfr.backends.cuda.driver import CUDA
        return CUDA().device_count()

    return 1


# Backward-compatible aliases.
_model_meta_path = model_meta_path
_resolve_model_path = resolve_model_path
_save_metadata = save_metadata
_load_metadata = load_metadata
_activation_from_name = activation_from_name

