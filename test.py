from math import pi
from equations_for_prob_measuring_state import (
    static_probability_equation_for_measuring_zero_no_complex,
    equation_for_kraus_probabilities_no_complex,
    static_probability_equation_for_measuring_zero_no_complex,
)


def test_equation_for_kraus_probabilities_no_complex():
    assert (
        round(
            equation_for_kraus_probabilities_no_complex(
                theta=0.0,
                eplison=-1.0,
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
            equation_for_kraus_probabilities_no_complex(
                theta=pi / 2,
                eplison=-289 * pi,
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
            equation_for_kraus_probabilities_no_complex(
                theta=pi, eplison=-9 / 5, x=-12 / 5, y=-1 / 2, z=39 / 10
            ),
            4,
        )
        == 0.0
    )


def test_whole_equation_for_probability_of_measuring_one_no_complex():
    assert (
        round(
            static_probability_equation_for_measuring_zero_no_complex(
                theta=0.0,
                eplison=1 / 2,
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
            static_probability_equation_for_measuring_zero_no_complex(
                theta=pi / 2, eplison=7 / 10, x=0.467884, y=3 / 34, z=10 / 51
            ),
            4,
        )
        == 0.5214
    )

    assert (
        round(
            static_probability_equation_for_measuring_zero_no_complex(
                theta=pi, eplison=-2 / 5, x=0.012232, y=0.0, z=61 / 102
            ),
            4,
        )
        == 0.0549
    )


def test_static_probability_of_measuring_zero_no_complex():
    assert not (
        round(
            static_probability_equation_for_measuring_zero_no_complex(
                theta=0.0,
                eplison=-2.60709,
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
                eplison=-2.60709,
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
                eplison=11.3,
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
                eplison=11.3,
                mu=-69.4,
                x=2.29807,
                y=31.3,
                z=-66.3,
            ),
            4,
        )
        == 0.5
    )


if __name__ == "__main__":
    test_equation_for_kraus_probabilities_no_complex()
    test_whole_equation_for_probability_of_measuring_one_no_complex()
    test_static_probability_of_measuring_zero_no_complex()
