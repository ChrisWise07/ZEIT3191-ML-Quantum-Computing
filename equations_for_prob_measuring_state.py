from numba import njit
import numpy as np


@njit(cache=True)
def probability_of_measuring_zero_given_ground_state(
    mu: float, epsilon: float
) -> np.complex64:
    """
    Probability of measuring zero given ground state.

    Args:
        mu: Parameter mu.
        epsilon: Parameter epsilon.
    Returns:
        Probability of measuring zero given ground state.
    """
    return (
        -(epsilon**2)
        + 2
        * epsilon
        * mu
        * (epsilon * mu + np.sqrt((epsilon**2 - 1) * (mu**2 - 1)))
        - mu**2
        + 1
    )


@njit(cache=True)
def probability_of_measuring_one_given_ground_state(
    nu: float, epsilon: float
) -> np.complex64:
    """
    Probability of measuring one given ground state.

    Args:
        mu: Parameter mu.
        epsilon: Parameter epsilon.
    Returns:
        Probability of measuring one given ground state.
    """
    return (
        epsilon**2
        + 2
        * epsilon
        * nu
        * (-epsilon * nu + np.sqrt((epsilon**2 - 1) * (nu**2 - 1)))
        + nu**2
    )


@njit(cache=True)
def probability_of_measuring_zero_given_excited_state(
    tau: float, mu: float
) -> np.complex64:
    """
    Probability of measuring zero given excited state.

    Args:
        tau: Parameter tau.
        mu: Parameter mu.

    Returns:
        Probability of measuring zero given excited state.
    """

    return (
        mu**2
        + 2 * mu * tau * (-mu * tau + np.sqrt((mu**2 - 1) * (tau**2 - 1)))
        + tau**2
    )


@njit(cache=True)
def probability_of_measuring_one_given_excited_state(
    tau: float, nu: float
) -> np.complex64:
    """
    Probability of measuring zero given excited state.

    Args:
        tau: Parameter tau.
        nu: Parameter nu.

    Returns:
        Probability of measuring one given excited state.
    """
    return (
        -(nu**2)
        + 2 * nu * tau * (nu * tau + np.sqrt((nu**2 - 1) * (tau**2 - 1)))
        - tau**2
        + 1
    )
