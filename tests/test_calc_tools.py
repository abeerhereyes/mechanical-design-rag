import math

import pytest

from src.calc_tools import (
    bearing_l10_life_hours,
    bolt_preload,
    spring_rate,
    tensile_stress_area_metric,
)


def test_spring_rate_matches_documented_example():
    assert spring_rate(79300, 3, 25, 8) == pytest.approx(6.4233, rel=1e-3)


def test_metric_stress_area_and_preload():
    area = tensile_stress_area_metric(10, 1.5)
    assert area == pytest.approx(57.99, rel=1e-3)
    result = bolt_preload(area, 827, reused=True)
    assert result["preload_N"] == pytest.approx(area * 827 * 0.75)


def test_bearing_life_uses_correct_exponent():
    result = bearing_l10_life_hours(15000, 3000, 1800, "ball")
    assert result["L10_million_rev"] == pytest.approx(125)
    assert result["L10_hours"] == pytest.approx(125e6 / (60 * 1800))


@pytest.mark.parametrize(
    "call",
    [
        lambda: spring_rate(79300, 0, 25, 8),
        lambda: tensile_stress_area_metric(1, 2),
        lambda: bearing_l10_life_hours(10, 0, 100),
    ],
)
def test_invalid_inputs_are_rejected(call):
    with pytest.raises(ValueError):
        call()
