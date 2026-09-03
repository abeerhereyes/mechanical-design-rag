import math

import pytest

from src.calculations import (
    FormulaDefinition,
    FormulaRegistry,
    FormulaReviewError,
    FormulaValidationError,
    OutputDefinition,
    ReviewState,
    SourceReference,
    VariableDefinition,
    execute,
    get_formula,
    list_formulas,
)


def test_registry_exposes_typed_provenance_for_every_course():
    formulas = list_formulas()
    assert {"mechanical-design", "aerodynamics", "qrm"} == {
        formula.course_id for formula in formulas
    }
    assert all(formula.source.document for formula in formulas)
    assert all(formula.source.pages for formula in formulas)
    assert all(formula.variables and formula.outputs for formula in formulas)
    assert all(formula.assumptions for formula in formulas)
    assert all(formula.review_state == ReviewState.REVIEWED for formula in formulas)


def test_course_filter_and_formula_lookup():
    aero = list_formulas(course_id="aerodynamics")
    assert aero
    assert all(formula.formula_id.startswith("aero.") for formula in aero)
    formula = get_formula("aero.kutta_joukowski_cylinder")
    assert formula.discrepancy_warning
    assert "counter-clockwise" in formula.discrepancy_warning


def test_registry_rejects_unreviewed_formula_execution():
    local = FormulaRegistry()
    local.register(
        FormulaDefinition(
            formula_id="test.draft",
            course_id="test",
            title="Draft",
            equation="y=x",
            source=SourceReference("test source", ("1",)),
            variables=(VariableDefinition("x", "input"),),
            outputs=(OutputDefinition("y", "output"),),
            assumptions=("Test only",),
            review_state=ReviewState.DRAFT,
            executor=lambda x: {"y": x},
        )
    )
    with pytest.raises(FormulaReviewError, match="only reviewed"):
        local.execute("test.draft", {"x": 1})


@pytest.mark.parametrize(
    ("inputs", "units", "message"),
    [
        (
            {"shear_modulus_mpa": 79300, "wire_diameter_mm": 3,
             "mean_diameter_mm": 25, "active_coils": 8},
            {"shear_modulus_mpa": "GPa", "wire_diameter_mm": "mm",
             "mean_diameter_mm": "mm"},
            "Invalid unit",
        ),
        (
            {"shear_modulus_mpa": 79300, "wire_diameter_mm": 3,
             "mean_diameter_mm": 25, "active_coils": 8},
            {"wire_diameter_mm": "mm", "mean_diameter_mm": "mm"},
            "Unit required",
        ),
        (
            {"shear_modulus_mpa": math.inf, "wire_diameter_mm": 3,
             "mean_diameter_mm": 25, "active_coils": 8},
            {"shear_modulus_mpa": "MPa", "wire_diameter_mm": "mm",
             "mean_diameter_mm": "mm"},
            "finite",
        ),
    ],
)
def test_execution_rejects_bad_units_and_values(inputs, units, message):
    with pytest.raises(FormulaValidationError, match=message):
        execute("mechanical.helical_spring_rate", inputs, units)


def test_unknown_and_missing_inputs_are_rejected():
    with pytest.raises(FormulaValidationError, match="Unknown inputs"):
        execute(
            "aero.pressure_coefficient",
            {"local_speed_m_per_s": 2, "freestream_speed_m_per_s": 3, "extra": 1},
            {
                "local_speed_m_per_s": "m/s",
                "freestream_speed_m_per_s": "m/s",
            },
        )
    with pytest.raises(FormulaValidationError, match="Missing required"):
        execute(
            "aero.pressure_coefficient",
            {"local_speed_m_per_s": 2},
            {"local_speed_m_per_s": "m/s"},
        )


def test_mechanical_port_matches_existing_result():
    result = execute(
        "mechanical.helical_spring_rate",
        {
            "shear_modulus_mpa": 79300,
            "wire_diameter_mm": 3,
            "mean_diameter_mm": 25,
            "active_coils": 8,
        },
        {
            "shear_modulus_mpa": "MPa",
            "wire_diameter_mm": "mm",
            "mean_diameter_mm": "mm",
        },
    )
    assert result.values["spring_rate"] == pytest.approx(6.4233, rel=1e-3)
    assert result.units["spring_rate"] == "N/mm"
