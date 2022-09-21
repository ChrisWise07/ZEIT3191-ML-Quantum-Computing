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


@njit(cache=True)
def whole_equation_for_probability_of_measuring_one(
    theta: float,
    phi: float,
    eplison: float,
    nu: float,
    mu: float,
    tau: float,
    kxtheta: float,
    kytheta: float,
    kztheta: float,
    kxphi: float,
    kyphi: float,
    kzphi: float,
) -> np.complex64:
    """
    Probability of measuring zero given excited state.

    Args:
        tau: Parameter tau.
        nu: Parameter nu.

    Returns:
        Probability of measuring one given excited state.
    """
    return 0.5 * (
        1
        - np.cos(theta + eplison)
        * np.cos(nu)
        * (
            2
            * (
                (np.sin((theta + eplison) / 2) ** 2) * (kxtheta + kytheta)
                + (np.sin((phi + mu) / 2) ** 2) * (kxphi + kyphi)
            )
            - 1
        )
        + np.sin(theta + eplison)
        * np.sin(nu)
        * (
            np.cos(phi + mu + tau)
            * (
                1
                - (
                    (np.sin((theta + eplison) / 2) ** 2)
                    * (2 * kztheta + kxtheta + kytheta)
                )
                - ((np.sin((phi + mu) / 2) ** 2) * (2 * kzphi + kxphi + kyphi))
            )
            + np.cos(phi + mu - tau)
            * (
                ((np.sin((theta + eplison) / 2) ** 2) * (kxtheta - kytheta))
                + ((np.sin((phi + mu) / 2) ** 2) * (kxphi - kyphi))
            )
        )
    )


@njit(cache=True)
def equation_for_kraus_probabilities(
    theta: float,
    phi: float,
    eplison: float,
    mu: float,
    kxtheta: float,
    kytheta: float,
    kztheta: float,
    kitheta: float,
    kxphi: float,
    kyphi: float,
    kzphi: float,
    kiphi: float,
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
        ((np.sin((theta + eplison) / 2) ** 2) * (kxtheta + kytheta + kztheta))
        + ((np.sin((phi + mu) / 2) ** 2) * (kxphi + kyphi + kzphi))
        + ((np.cos((theta + eplison) / 2) ** 2) * (kitheta))
        + ((np.cos((phi + mu) / 2) ** 2) * (kiphi))
    )


@njit(cache=True)
def whole_equation_for_probability_of_measuring_one_no_complex(
    theta: float,
    eplison: float,
    x: float,
    y: float,
    z: float,
) -> float:
    """
    Probability of measuring zero given excited state.
    """
    return (
        (1 / 2)
        + 0.48785
        * (
            1
            - x
            - y
            + (x + y)
            * (np.cos(1 / 2 * (eplison + 2 * theta - eplison * np.cos(theta))))
        )
        * np.cos(theta + eplison * (np.sin(theta / 2) ** 2))
        + 0.109555
        * (
            1
            - y
            - z
            + (y + z)
            * (np.cos(1 / 2 * (eplison + 2 * theta - eplison * np.cos(theta))))
        )
        * np.sin(theta + eplison * (np.sin(theta / 2) ** 2))
    )


@njit(cache=True)
def equation_for_kraus_probabilities_no_complex(
    theta: float,
    eplison: float,
    x: float,
    y: float,
    z: float,
    i: float,
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
        x
        + y
        + z
        + i
        + (i - x - y - z) * np.cos(theta + eplison * (np.sin(theta / 2) ** 2))
    )
