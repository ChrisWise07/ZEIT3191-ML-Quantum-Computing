from numba import njit


@njit(cache=True)
def probability_of_measuring_zero_given_ground_state(
    mu: float, epsilon: float
) -> float:
    """
    Probability of measuring zero given ground state.

    Args:
        mu: Parameter mu.
        epsilon: Parameter epsilon.
    Returns:
        Probability of measuring zero given ground state.
    """
    return (1 - mu - epsilon) ** 2 * (1 + 2 * mu * (mu - 1))


@njit(cache=True)
def probability_of_measuring_one_given_ground_state(
    nu: float, epsilon: float
) -> float:
    """
    Probability of measuring one given ground state.

    Args:
        mu: Parameter mu.
        epsilon: Parameter epsilon.
    Returns:
        Probability of measuring one given ground state.
    """
    return (2 * nu * (nu - 1) + 1) * (-2 * epsilon * nu + epsilon + nu) ** 2


@njit(cache=True)
def probability_of_measuring_zero_given_excited_state(
    tau: float, mu: float
) -> float:
    """
    Probability of measuring zero given excited state.

    Args:
        tau: Parameter tau.
        mu: Parameter mu.

    Returns:
        Probability of measuring zero given excited state.
    """

    return (mu + tau) ** 2 * (2 * mu * (mu - 1) + 1)


@njit(cache=True)
def probability_of_measuring_one_given_excited_state(
    tau: float, nu: float
) -> float:
    """
    Probability of measuring zero given excited state.

    Args:
        tau: Parameter tau.
        nu: Parameter nu.

    Returns:
        Probability of measuring one given excited state.
    """
    return (2 * nu * (nu - 1) + 1) * (2 * nu * tau - nu - tau + 1) ** 2
