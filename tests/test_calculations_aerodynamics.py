import math

import pytest

from src.calculations import FormulaValidationError, execute


def test_force_decomposition_at_zero_angle():
    result = execute(
        "aero.force_decomposition",
        {"normal_force_n": 100, "axial_force_n": 20, "angle_of_attack_rad": 0},
        {
            "normal_force_n": "N",
            "axial_force_n": "N",
            "angle_of_attack_rad": "rad",
        },
    )
    assert result.values == pytest.approx({"lift": 100, "drag": 20})


def test_source_and_vortex_velocity_sign_conventions():
    source = execute(
        "aero.point_source_velocity",
        {"x_m": 2, "y_m": 0, "strength_m2_per_s": 4 * math.pi},
        {"x_m": "m", "y_m": "m", "strength_m2_per_s": "m^2/s"},
    )
    assert source.values == pytest.approx({"velocity_x": 1, "velocity_y": 0})

    vortex = execute(
        "aero.point_vortex_velocity",
        {"x_m": 1, "y_m": 0, "circulation_m2_per_s": 2 * math.pi},
        {"x_m": "m", "y_m": "m", "circulation_m2_per_s": "m^2/s"},
    )
    assert vortex.values == pytest.approx({"velocity_x": 0, "velocity_y": 1})
    assert vortex.warnings


def test_doublet_velocity_uses_declared_potential_convention():
    result = execute(
        "aero.doublet_velocity",
        {"x_m": 1, "y_m": 0, "strength_m3_per_s": 2 * math.pi},
        {"x_m": "m", "y_m": "m", "strength_m3_per_s": "m^3/s"},
    )
    assert result.values == pytest.approx({"velocity_x": -1, "velocity_y": 0})


def test_cylinder_surface_flow_and_pressure_coefficient():
    velocity = execute(
        "aero.cylinder_flow_velocity",
        {
            "freestream_speed_m_per_s": 10,
            "cylinder_radius_m": 1,
            "radial_position_m": 1,
            "theta_rad": math.pi / 2,
        },
        {
            "freestream_speed_m_per_s": "m/s",
            "cylinder_radius_m": "m",
            "radial_position_m": "m",
            "theta_rad": "rad",
        },
    )
    assert velocity.values["radial_velocity"] == pytest.approx(0, abs=1e-12)
    assert velocity.values["tangential_velocity"] == pytest.approx(-20)

    cp = execute(
        "aero.pressure_coefficient",
        {"local_speed_m_per_s": 20, "freestream_speed_m_per_s": 10},
        {"local_speed_m_per_s": "m/s", "freestream_speed_m_per_s": "m/s"},
    )
    assert cp.values["pressure_coefficient"] == pytest.approx(-3)


def test_kutta_joukowski_counter_clockwise_positive_sign():
    result = execute(
        "aero.kutta_joukowski_cylinder",
        {
            "density_kg_per_m3": 1.2,
            "freestream_speed_m_per_s": 10,
            "circulation_m2_per_s": 3,
        },
        {
            "density_kg_per_m3": "kg/m^3",
            "freestream_speed_m_per_s": "m/s",
            "circulation_m2_per_s": "m^2/s",
        },
    )
    assert result.values["lift_per_span"] == pytest.approx(-36)
    assert result.values["drag_per_span"] == 0
    assert "counter-clockwise" in result.warnings[0]


@pytest.mark.parametrize(
    ("formula_id", "inputs", "units"),
    [
        (
            "aero.point_source_velocity",
            {"x_m": 0, "y_m": 0, "strength_m2_per_s": 1},
            {"x_m": "m", "y_m": "m", "strength_m2_per_s": "m^2/s"},
        ),
        (
            "aero.cylinder_flow_velocity",
            {
                "freestream_speed_m_per_s": 10,
                "cylinder_radius_m": 2,
                "radial_position_m": 1,
                "theta_rad": 0,
            },
            {
                "freestream_speed_m_per_s": "m/s",
                "cylinder_radius_m": "m",
                "radial_position_m": "m",
                "theta_rad": "rad",
            },
        ),
    ],
)
def test_invalid_flow_locations_are_rejected(formula_id, inputs, units):
    with pytest.raises(FormulaValidationError):
        execute(formula_id, inputs, units)
