from math import pi
from equations_for_prob_measuring_state import (
    static_kraus_probability_bounding_equation,
    trig_kraus_probability_bounding_equation,
    trig_probability_equation_for_measuring_zero_no_complex,
    static_probability_equation_for_measuring_zero_no_complex,
    static_kraus_probability_bounding_equation,
    partial_solved_trig_probability_equation_for_measuring_zero_no_complex,
    partial_solved_trig_equation_for_kraus_probabilities_no_complex,
    state_depedent_small_theta_no_complex_prob_equation,
    state_dependent_small_theta_no_complex_kraus_bounding_equation,
)


def test_partial_solved_trig_equation_for_kraus_probabilities_no_complex():
    assert not (
        round(
            trig_kraus_probability_bounding_equation(
                theta=0.0, epsilon=-1.1, x=0.5, y=-9.3, z=3.2, l=1.0
            ),
            4,
        )
        == 2
    )

    assert (
        round(
            trig_kraus_probability_bounding_equation(
                theta=0.0, epsilon=-1.1, x=0.5, y=-9.3, z=3.2, l=1.0
            ),
            4,
        )
        == 0.0000
    )

    assert (
        round(
            trig_kraus_probability_bounding_equation(
                theta=0.0, epsilon=-5.3, x=0.6, y=-10.0, z=-2.8, l=1.0
            ),
            4,
        )
        == 0.0000
    )

    assert (
        round(
            trig_kraus_probability_bounding_equation(
                theta=pi / 2, epsilon=171.444, x=-0.6, y=-3.2, z=4.4, l=1.04872
            ),
            3,
        )
        == 0.000
    )

    assert (
        round(
            trig_kraus_probability_bounding_equation(
                theta=pi / 4, epsilon=659.653, x=0.5, y=-9.3, z=9.8, l=-3.9
            ),
            3,
        )
        == 0.000
    )


def test_trig_probability_of_measuring_zero_no_complex():
    assert not (
        round(
            trig_probability_equation_for_measuring_zero_no_complex(
                theta=0.0,
                epsilon=86.1,
                mu=-44.2037,
                x=-23.3,
                y=23.9,
                z=55.1,
            ),
            4,
        )
        == 0.0122
    )

    assert (
        round(
            trig_probability_equation_for_measuring_zero_no_complex(
                theta=0.0,
                epsilon=86.1,
                mu=-44.2037,
                x=-23.3,
                y=23.9,
                z=55.1,
            ),
            4,
        )
        == 0.9878
    )

    assert (
        round(
            trig_probability_equation_for_measuring_zero_no_complex(
                theta=0.0,
                epsilon=11.3,
                mu=-69.3364,
                x=-87.2,
                y=31.3,
                z=-66.3,
            ),
            4,
        )
        == 0.9878
    )

    assert (
        round(
            trig_probability_equation_for_measuring_zero_no_complex(
                theta=pi / 2,
                epsilon=86.1,
                mu=-45.4,
                x=336.763,
                y=23.9,
                z=55.1,
            ),
            4,
        )
        == 0.5000
    )


def test_static_probability_of_measuring_zero_no_complex():
    assert not (
        round(
            static_probability_equation_for_measuring_zero_no_complex(
                theta=0.0,
                epsilon=-2.60709,
                mu=0.725278,
                x=0.0402137,
                y=-0.0903284,
                z=3.08871,
            ),
            4,
        )
        == 0.0122
    )

    assert (
        round(
            static_probability_equation_for_measuring_zero_no_complex(
                theta=0.0,
                epsilon=-2.60709,
                mu=0.725278,
                x=0.0402137,
                y=-0.0903284,
                z=3.08871,
            ),
            4,
        )
        == 0.99
    )

    assert (
        round(
            static_probability_equation_for_measuring_zero_no_complex(
                theta=0.0,
                epsilon=11.3,
                mu=-69.4,
                x=0.594659,
                y=31.3,
                z=-66.3,
            ),
            4,
        )
        == 0.99
    )

    assert (
        round(
            static_probability_equation_for_measuring_zero_no_complex(
                theta=pi,
                epsilon=11.3,
                mu=-69.4,
                x=2.29807,
                y=31.3,
                z=-66.3,
            ),
            4,
        )
        == 0.5
    )


def test_static_kraus_bonding_equation() -> None:
    assert (
        round(
            static_kraus_probability_bounding_equation(
                theta=0.0, x=0.5, y=0.5, z=0, l=0
            ),
            4,
        )
        == 0.0
    )

    assert (
        round(
            static_kraus_probability_bounding_equation(
                theta=0.0, x=0.25, y=0.25, z=0.25, l=0.25
            ),
            4,
        )
        == 0.0
    )


def test_partial_solved_partial_solved_trig_equation_for_kraus_probabilities_no_complex():
    assert (
        round(
            partial_solved_trig_equation_for_kraus_probabilities_no_complex(
                theta=0.0,
                epsilon=-1.0,
                x=1.0,
                y=1.0,
                z=48 / 101,
            ),
            4,
        )
        == 0.0
    )

    assert (
        round(
            partial_solved_trig_equation_for_kraus_probabilities_no_complex(
                theta=pi / 2,
                epsilon=-289 * pi,
                x=-3 / 10,
                y=-21 / 5,
                z=-23 / 10,
            ),
            4,
        )
        == 0.0
    )

    assert (
        round(
            partial_solved_trig_equation_for_kraus_probabilities_no_complex(
                theta=pi, epsilon=-9 / 5, x=-12 / 5, y=-1 / 2, z=39 / 10
            ),
            4,
        )
        == 0.0
    )


def test_partial_solved_equation_for_probability_of_measuring_one_no_complex():
    assert (
        round(
            partial_solved_trig_probability_equation_for_measuring_zero_no_complex(
                theta=0.0,
                epsilon=1 / 2,
                x=0.0310517,
                y=0.0,
                z=11 / 102,
            ),
            4,
        )
        == 0.9878
    )
    assert (
        round(
            partial_solved_trig_probability_equation_for_measuring_zero_no_complex(
                theta=pi / 2, epsilon=7 / 10, x=0.467884, y=3 / 34, z=10 / 51
            ),
            4,
        )
        == 0.5214
    )

    assert (
        round(
            partial_solved_trig_probability_equation_for_measuring_zero_no_complex(
                theta=pi, epsilon=-2 / 5, x=0.012232, y=0.0, z=61 / 102
            ),
            4,
        )
        == 0.0549
    )


def test_state_depedent_small_theta_no_complex_prob_equation() -> None:
    assert (
        round(
            state_depedent_small_theta_no_complex_prob_equation(
                theta=0.0,
                epsilon=5.0,
                x=5.0,
                y=5.0,
                z=5,
            ),
            5,
        )
        == 0.98785
    )

    assert (
        round(
            state_depedent_small_theta_no_complex_prob_equation(
                theta=0.0,
                epsilon=0.0,
                x=0.0,
                y=0.0,
                z=0.0,
            ),
            5,
        )
        == 0.98785
    )

    assert (
        round(
            state_depedent_small_theta_no_complex_prob_equation(
                theta=pi / 2,
                epsilon=-0.700509,
                x=1.23549,
                y=0.598183,
                z=-2.24748,
            ),
            5,
        )
        == 0.55555
    )

    assert (
        round(
            state_depedent_small_theta_no_complex_prob_equation(
                theta=pi, epsilon=0.367796, x=2.81639, y=-2.7732, z=2.20524
            ),
            5,
        )
        == 0.00005
    )


def test_state_dependent_small_theta_no_complex_kraus_bounding_equation() -> None:
    assert (
        round(
            state_dependent_small_theta_no_complex_kraus_bounding_equation(
                theta=0.0,
                epsilon=5.0,
                x=5.0,
                y=5.0,
                z=5,
            ),
            5,
        )
        == 0.0
    )

    assert (
        round(
            state_dependent_small_theta_no_complex_kraus_bounding_equation(
                theta=0.0,
                epsilon=0.0,
                x=0.0,
                y=0.0,
                z=0.0,
            ),
            5,
        )
        == 0.0
    )

    assert (
        round(
            state_dependent_small_theta_no_complex_kraus_bounding_equation(
                theta=pi / 2,
                epsilon=-2.22144,
                x=-2.95459,
                y=3.16666,
                z=1.37721,
            ),
            5,
        )
        == 0.0
    )

    assert (
        round(
            state_dependent_small_theta_no_complex_kraus_bounding_equation(
                theta=pi, epsilon=-3.14159, x=0.612319, y=0.488268, z=2.34799
            ),
            5,
        )
        == 0.0
    )


if __name__ == "__main__":
    test_trig_probability_of_measuring_zero_no_complex()
    test_partial_solved_trig_equation_for_kraus_probabilities_no_complex()
    test_static_probability_of_measuring_zero_no_complex()
    test_static_kraus_bonding_equation()
    test_partial_solved_partial_solved_trig_equation_for_kraus_probabilities_no_complex()
    test_partial_solved_equation_for_probability_of_measuring_one_no_complex()
    test_state_depedent_small_theta_no_complex_prob_equation()
    test_state_dependent_small_theta_no_complex_kraus_bounding_equation()
