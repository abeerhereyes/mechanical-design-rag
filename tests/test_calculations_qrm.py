import math

import pytest

from src.calculations import FormulaValidationError, execute


def test_distribution_probabilities():
    normal = execute(
        "qrm.normal_cdf",
        {"x": 0, "mean": 0, "standard_deviation": 1},
        {"x": "observation", "mean": "observation",
         "standard_deviation": "observation"},
    )
    assert normal.values["probability"] == pytest.approx(0.5)

    binomial = execute(
        "qrm.binomial_probability",
        {"trials": 4, "successes": 2, "event_probability": 0.5},
    )
    assert binomial.values["probability"] == pytest.approx(0.375)

    poisson = execute(
        "qrm.poisson_probability",
        {"event_count": 0, "mean_count": 2},
    )
    assert poisson.values["probability"] == pytest.approx(math.exp(-2))


def test_binomial_and_hypergeometric_oc_probabilities():
    binomial = execute(
        "qrm.acceptance_oc_binomial",
        {"sample_size": 10, "acceptance_number": 0, "fraction_defective": 0.1},
    )
    assert binomial.values["acceptance_probability"] == pytest.approx(0.9**10)

    hypergeometric = execute(
        "qrm.acceptance_oc_hypergeometric",
        {
            "lot_size": 10,
            "defective_units": 2,
            "sample_size": 3,
            "acceptance_number": 0,
        },
    )
    assert hypergeometric.values["acceptance_probability"] == pytest.approx(
        math.comb(8, 3) / math.comb(10, 3)
    )


def test_rectifying_aoq_ati_asn():
    result = execute(
        "qrm.rectifying_sampling_performance",
        {
            "lot_size": 1000,
            "sample_size": 50,
            "fraction_defective": 0.02,
            "acceptance_probability": 0.8,
        },
    )
    assert result.values["average_outgoing_quality"] == pytest.approx(0.0152)
    assert result.values["average_total_inspection"] == pytest.approx(240)
    assert result.values["average_sample_number"] == 50


def test_arl_and_xbar_control_limits():
    arl = execute("qrm.average_run_length", {"signal_probability": 0.0027})
    assert arl.values["average_run_length"] == pytest.approx(1 / 0.0027)

    limits = execute(
        "qrm.xbar_control_limits",
        {
            "process_mean": 100,
            "process_standard_deviation": 4,
            "sample_size": 16,
            "sigma_multiplier": 3,
        },
        {
            "process_mean": "observation",
            "process_standard_deviation": "observation",
        },
    )
    assert limits.values == pytest.approx(
        {"lower_control_limit": 97, "center_line": 100, "upper_control_limit": 103}
    )


def test_p_chart_returns_raw_and_clipped_limits_with_warning():
    result = execute(
        "qrm.p_control_limits",
        {
            "average_fraction_nonconforming": 0.01,
            "sample_size": 100,
            "sigma_multiplier": 3,
        },
    )
    assert result.values["raw_lower_control_limit"] < 0
    assert result.values["lower_control_limit"] == 0
    assert result.warnings


def test_exponential_and_weibull_reliability():
    exponential = execute(
        "qrm.exponential_reliability",
        {"time": 100, "failure_rate_per_time": 0.01},
        {"time": "time", "failure_rate_per_time": "1/time"},
    )
    assert exponential.values["reliability"] == pytest.approx(math.exp(-1))
    assert exponential.values["mean_time_to_failure"] == pytest.approx(100)

    weibull = execute(
        "qrm.weibull_reliability",
        {"time": 10, "scale_time": 10, "shape": 2},
        {"time": "time", "scale_time": "time"},
    )
    assert weibull.values["reliability"] == pytest.approx(math.exp(-1))
    assert weibull.values["hazard_rate"] == pytest.approx(0.2)
    assert weibull.values["mean_time_to_failure"] == pytest.approx(
        10 * math.gamma(1.5)
    )


@pytest.mark.parametrize(
    ("formula_id", "inputs"),
    [
        (
            "qrm.acceptance_oc_binomial",
            {"sample_size": 5, "acceptance_number": 6, "fraction_defective": 0.1},
        ),
        (
            "qrm.acceptance_oc_hypergeometric",
            {"lot_size": 5, "defective_units": 1,
             "sample_size": 6, "acceptance_number": 0},
        ),
        (
            "qrm.rectifying_sampling_performance",
            {"lot_size": 10, "sample_size": 11,
             "fraction_defective": 0.1, "acceptance_probability": 0.8},
        ),
        (
            "qrm.average_run_length",
            {"signal_probability": 0},
        ),
    ],
)
def test_invalid_qrm_inputs_are_rejected(formula_id, inputs):
    with pytest.raises(FormulaValidationError):
        execute(formula_id, inputs)
