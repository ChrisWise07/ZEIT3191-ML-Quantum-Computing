from numba import njit
import numpy as np


@njit(cache=True)
def trig_probability_equation_for_measuring_zero_no_complex(
    theta: float,
    epsilon: float,
    mu: float,
    x: float,
    y: float,
    z: float,
) -> float:
    """
    Probability of measuring zero given excited state.
    """
    return (1 / 2) * (
        1
        + np.cos(mu)
        * (
            1
            - x
            - y
            + (x + y)
            * (np.cos(1 / 2 * (epsilon + 2 * theta - epsilon * np.cos(theta))))
        )
        * np.cos(theta + epsilon * (np.sin(theta / 2) ** 2))
        + (
            1
            - y
            - z
            + (y + z)
            * (np.cos(1 / 2 * (epsilon + 2 * theta - epsilon * np.cos(theta))))
        )
        * np.sin(mu)
        * np.sin(theta + epsilon * (np.sin(theta / 2) ** 2))
    )


@njit(cache=True)
def trig_kraus_probability_bounding_equation(
    theta: float,
    epsilon: float,
    x: float,
    y: float,
    z: float,
    l: float,
) -> float:
    """
    Probability of measuring zero given excited state.

    Args:
        tau: Parameter tau.
        nu: Parameter nu.

    Returns:
        Probability of measuring one given excited state.
    """
    return (
        -2
        + l
        + x
        + y
        + z
        + (l - x - y - z)
        * np.cos(theta + epsilon * np.square(np.sin(theta / 2)))
    )


@njit(cache=True)
def static_probability_equation_for_measuring_zero_no_complex(
    theta: float,
    epsilon: float,
    mu: float,
    x: float,
    y: float,
    z: float,
) -> float:
    """ """
    return (1 / 2) * (
        1
        + (1 - 2 * x - 2 * y) * np.cos(epsilon + theta) * np.cos(mu)
        + (1 - 2 * y - 2 * z) * np.sin(epsilon + theta) * np.sin(mu)
    )


@njit(cache=True)
def static_kraus_probability_bounding_equation(
    theta: float,
    x: float,
    y: float,
    z: float,
    l: float,
) -> float:
    """ """
    return x + y + z + l - 1


@njit(cache=True)
def partial_solved_trig_probability_equation_for_measuring_zero_no_complex(
    theta: float,
    epsilon: float,
    x: float,
    y: float,
    z: float,
) -> float:
    """
    Probability of measuring zero given excited state.
    """
    return (
        (1 / 2)
        + 0.4878
        * (
            1
            - x
            - y
            + (x + y)
            * (np.cos(1 / 2 * (epsilon + 2 * theta - epsilon * np.cos(theta))))
        )
        * np.cos(theta + epsilon * (np.sin(theta / 2) ** 2))
        + 0.109778
        * (
            1
            - y
            - z
            + (y + z)
            * (np.cos(1 / 2 * (epsilon + 2 * theta - epsilon * np.cos(theta))))
        )
        * np.sin(theta + epsilon * (np.sin(theta / 2) ** 2))
    )


@njit(cache=True)
def partial_solved_trig_equation_for_kraus_probabilities_no_complex(
    theta: float,
    epsilon: float,
    x: float,
    y: float,
    z: float,
) -> float:
    """
    Probability of measuring zero given excited state.
    Args:
        tau: Parameter tau.
        nu: Parameter nu.
    Returns:
        Probability of measuring one given excited state.
    """
    return (-1 + x + y + z) * (
        -1 + np.cos(theta + epsilon * np.square(np.sin(theta / 2)))
    )
