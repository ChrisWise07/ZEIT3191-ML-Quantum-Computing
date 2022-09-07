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


@njit(cache=True)
def return_init_np_array_for_single_qubit(
    theta: float, phi: float
) -> np.ndarray:
    """
    Return

    Args:
        theta:
            The theta (in radians) to use for qubit init.
        phi:
            The phi (in radians) to use for qubit init.

    Returns:
        A numpy array of the form [
            cos(theta/2),
            sin(theta/2) * exp(i * phi)
        ].

    """
    return np.array(
        [
            np.cos(theta / 2),
            np.sin(theta / 2) * np.exp(1j * phi),
        ]
    )
