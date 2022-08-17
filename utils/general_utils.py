from numba import njit
import numpy as np


def file_handler(path, mode, func):
    try:
        with open(path, mode) as f:
            return func(f)
    except FileNotFoundError:
        return 0


@njit(cache=True)
def sin_with_complex(theta: float, phi: float) -> float:
    return np.sin(theta / 2) * (np.cos(phi))


@njit(cache=True)
def cos_with_real(theta: float) -> float:
    return np.cos(theta / 2)
