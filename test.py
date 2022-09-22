from math import pi
from equations_for_prob_measuring_state import (
    whole_equation_for_probability_of_measuring_one_no_complex,
    equation_for_kraus_probabilities_no_complex,
)


def test_equation_for_kraus_probabilities_no_complex():
    assert (
        equation_for_kraus_probabilities_no_complex(
            theta=0.0,
            eplison=-1.0,
            x=1.0,
            y=1.0,
            z=48 / 101,
        )
        == 0.0
    )

    assert (
        equation_for_kraus_probabilities_no_complex(
            theta=pi / 2,
            eplison=-289 * pi,
            x=-3 / 10,
            y=-21 / 5,
            z=-23 / 10,
        )
        == 0.0
    )

    assert (
        equation_for_kraus_probabilities_no_complex(
            theta=pi,
            eplison=-9 / 5,
            x=-12 / 5,
            y=-1 / 2,
            z=39 / 10,
        )
        == 0.0
    )


def test_whole_equation_for_probability_of_measuring_one_no_complex():
    assert (
        whole_equation_for_probability_of_measuring_one_no_complex(
            theta=0.0,
            eplison=1 / 2,
            x=0.0310517,
            y=0.0,
            z=11 / 102,
        )
        == 0.9878
    )
    assert (
        whole_equation_for_probability_of_measuring_one_no_complex(
            theta=pi / 2,
            eplison=7 / 10,
            x=0.467884,
            y=3 / 34,
            z=10 / 51,
        )
        == 0.5214
    )

    assert (
        whole_equation_for_probability_of_measuring_one_no_complex(
            theta=pi,
            eplison=-2 / 5,
            x=0.012232,
            y=0.0,
            z=61 / 102,
        )
        == 0.0549
    )


if __name__ == "__main__":
    test_equation_for_kraus_probabilities_no_complex()
    test_whole_equation_for_probability_of_measuring_one_no_complex()
