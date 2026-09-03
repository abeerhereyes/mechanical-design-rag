"""Reviewed Quantitative/Reliability Management formulas."""
import math
from typing import Dict, Tuple

from .registry import (
    FormulaDefinition,
    FormulaRegistry,
    FormulaValidationError,
    OutputDefinition,
    ReviewState,
    SourceReference,
    VariableDefinition,
)

COURSE_ID = "qrm"


def _source(document: str, pages: Tuple[str, ...], section: str) -> SourceReference:
    return SourceReference(document, pages, section)


def _number(
    name: str,
    description: str,
    unit: str = "1",
    minimum=None,
    maximum=None,
    minimum_exclusive: bool = False,
) -> VariableDefinition:
    return VariableDefinition(
        name,
        description,
        unit=unit,
        minimum=minimum,
        maximum=maximum,
        minimum_exclusive=minimum_exclusive,
    )


def _integer(
    name: str, description: str, minimum: int = 0
) -> VariableDefinition:
    return VariableDefinition(
        name,
        description,
        value_type="integer",
        minimum=minimum,
    )


def _out(name: str, description: str, unit: str = "1") -> OutputDefinition:
    return OutputDefinition(name, description, unit)


def _normal_cdf(x: float, mean: float, standard_deviation: float) -> Dict[str, float]:
    z = (x - mean) / standard_deviation
    return {"probability": 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))}


def _binomial_probability(n: int, successes: int, probability: float) -> float:
    return (
        math.comb(n, successes)
        * probability**successes
        * (1.0 - probability) ** (n - successes)
    )


def _binomial_exact(
    trials: int, successes: int, event_probability: float
) -> Dict[str, float]:
    if successes > trials:
        raise FormulaValidationError("successes cannot exceed trials")
    return {
        "probability": _binomial_probability(
            trials, successes, event_probability
        )
    }


def _poisson_exact(event_count: int, mean_count: float) -> Dict[str, float]:
    return {
        "probability": (
            math.exp(-mean_count)
            * mean_count**event_count
            / math.factorial(event_count)
        )
    }


def _binomial_oc(
    sample_size: int, acceptance_number: int, fraction_defective: float
) -> Dict[str, float]:
    if acceptance_number > sample_size:
        raise FormulaValidationError(
            "acceptance_number cannot exceed sample_size"
        )
    probability = sum(
        _binomial_probability(sample_size, defects, fraction_defective)
        for defects in range(acceptance_number + 1)
    )
    return {"acceptance_probability": min(1.0, max(0.0, probability))}


def _hypergeometric_oc(
    lot_size: int,
    defective_units: int,
    sample_size: int,
    acceptance_number: int,
) -> Dict[str, float]:
    if defective_units > lot_size:
        raise FormulaValidationError("defective_units cannot exceed lot_size")
    if sample_size > lot_size:
        raise FormulaValidationError("sample_size cannot exceed lot_size")
    if acceptance_number > sample_size:
        raise FormulaValidationError(
            "acceptance_number cannot exceed sample_size"
        )
    denominator = math.comb(lot_size, sample_size)
    largest = min(acceptance_number, defective_units, sample_size)
    smallest = max(0, sample_size - (lot_size - defective_units))
    probability = sum(
        math.comb(defective_units, defects)
        * math.comb(lot_size - defective_units, sample_size - defects)
        / denominator
        for defects in range(smallest, largest + 1)
    )
    return {"acceptance_probability": min(1.0, max(0.0, probability))}


def _sampling_performance(
    lot_size: int,
    sample_size: int,
    fraction_defective: float,
    acceptance_probability: float,
) -> Dict[str, float]:
    if sample_size > lot_size:
        raise FormulaValidationError("sample_size cannot exceed lot_size")
    return {
        "average_outgoing_quality": (
            fraction_defective
            * acceptance_probability
            * (lot_size - sample_size)
            / lot_size
        ),
        "average_total_inspection": (
            sample_size * acceptance_probability
            + lot_size * (1.0 - acceptance_probability)
        ),
        "average_sample_number": float(sample_size),
    }


def _arl(signal_probability: float) -> Dict[str, float]:
    return {"average_run_length": 1.0 / signal_probability}


def _xbar_limits(
    process_mean: float,
    process_standard_deviation: float,
    sample_size: int,
    sigma_multiplier: float,
) -> Dict[str, float]:
    margin = (
        sigma_multiplier * process_standard_deviation / math.sqrt(sample_size)
    )
    return {
        "lower_control_limit": process_mean - margin,
        "center_line": process_mean,
        "upper_control_limit": process_mean + margin,
    }


def _p_chart_limits(
    average_fraction_nonconforming: float,
    sample_size: int,
    sigma_multiplier: float,
) -> Dict[str, float]:
    margin = sigma_multiplier * math.sqrt(
        average_fraction_nonconforming
        * (1.0 - average_fraction_nonconforming)
        / sample_size
    )
    raw_lower = average_fraction_nonconforming - margin
    raw_upper = average_fraction_nonconforming + margin
    return {
        "raw_lower_control_limit": raw_lower,
        "lower_control_limit": max(0.0, raw_lower),
        "center_line": average_fraction_nonconforming,
        "upper_control_limit": min(1.0, raw_upper),
        "raw_upper_control_limit": raw_upper,
    }


def _exponential_reliability(
    time: float, failure_rate_per_time: float
) -> Dict[str, float]:
    reliability = math.exp(-failure_rate_per_time * time)
    return {
        "reliability": reliability,
        "failure_probability": 1.0 - reliability,
        "hazard_rate": failure_rate_per_time,
        "mean_time_to_failure": 1.0 / failure_rate_per_time,
    }


def _weibull_reliability(
    time: float, scale_time: float, shape: float
) -> Dict[str, float]:
    ratio = time / scale_time
    reliability = math.exp(-(ratio**shape))
    hazard = (shape / scale_time) * ratio ** (shape - 1.0)
    return {
        "reliability": reliability,
        "failure_probability": 1.0 - reliability,
        "hazard_rate": hazard,
        "mean_time_to_failure": scale_time * math.gamma(1.0 + 1.0 / shape),
    }


FORMULAS: Tuple[FormulaDefinition, ...] = (
    FormulaDefinition(
        "qrm.normal_cdf",
        COURSE_ID,
        "Normal cumulative probability",
        "P(X<=x)=Phi((x-mu)/sigma)",
        _source(
            "Quality_Reliability_Maintenance_ME3327E.pdf",
            ("4-8",),
            "Probability distributions — normal distribution",
        ),
        (
            _number("x", "Threshold", "observation"),
            _number("mean", "Distribution mean", "observation"),
            _number(
                "standard_deviation",
                "Distribution standard deviation",
                "observation",
                minimum=0.0,
                minimum_exclusive=True,
            ),
        ),
        (_out("probability", "Cumulative probability"),),
        ("Normal distribution", "All observation inputs use the same unit"),
        ReviewState.REVIEWED,
        _normal_cdf,
    ),
    FormulaDefinition(
        "qrm.binomial_probability",
        COURSE_ID,
        "Binomial exact probability",
        "P(X=x)=C(n,x)p^x(1-p)^(n-x)",
        _source(
            "Quality_Reliability_Maintenance_L6_Probability_Distribution.pdf",
            ("6-7",),
            "Probability distributions — binomial distribution",
        ),
        (
            _integer("trials", "Number of independent trials"),
            _integer("successes", "Exact number of successes"),
            _number(
                "event_probability",
                "Constant success probability",
                minimum=0.0,
                maximum=1.0,
            ),
        ),
        (_out("probability", "Exact probability"),),
        ("Independent Bernoulli trials", "Constant event probability"),
        ReviewState.REVIEWED,
        _binomial_exact,
    ),
    FormulaDefinition(
        "qrm.poisson_probability",
        COURSE_ID,
        "Poisson exact probability",
        "P(X=x)=exp(-lambda)lambda^x/x!",
        _source(
            "Quality_Reliability_Maintenance_L6_Probability_Distribution.pdf",
            ("9",),
            "Probability distributions — Poisson distribution",
        ),
        (
            _integer("event_count", "Exact event count"),
            _number(
                "mean_count",
                "Expected event count",
                minimum=0.0,
                minimum_exclusive=True,
            ),
        ),
        (_out("probability", "Exact probability"),),
        ("Independent events at a constant average rate",),
        ReviewState.REVIEWED,
        _poisson_exact,
    ),
    FormulaDefinition(
        "qrm.acceptance_oc_binomial",
        COURSE_ID,
        "Single-sampling OC probability (binomial)",
        "P_a=sum[x=0..c] C(n,x)p^x(1-p)^(n-x)",
        _source(
            "Acceptance_Sampling_OC_Curve_ME327E.pdf",
            ("2-7",),
            "Acceptance sampling — operating characteristic curve",
        ),
        (
            _integer("sample_size", "Number sampled", minimum=1),
            _integer("acceptance_number", "Maximum accepted defect count"),
            _number(
                "fraction_defective",
                "Incoming fraction defective",
                minimum=0.0,
                maximum=1.0,
            ),
        ),
        (_out("acceptance_probability", "Probability lot is accepted"),),
        (
            "Single-sampling attributes plan",
            "Binomial approximation: effectively infinite or large lot",
            "Independent classifications with constant defect probability",
        ),
        ReviewState.REVIEWED,
        _binomial_oc,
    ),
    FormulaDefinition(
        "qrm.acceptance_oc_hypergeometric",
        COURSE_ID,
        "Finite-lot single-sampling OC probability",
        "P_a=sum[x=0..c] C(D,x)C(N-D,n-x)/C(N,n)",
        _source(
            "Acceptance_Sampling_OC_Curve_ME327E.pdf",
            ("2-7",),
            "Acceptance sampling — exact finite-lot OC curve",
        ),
        (
            _integer("lot_size", "Finite lot size", minimum=1),
            _integer("defective_units", "Defective units in lot"),
            _integer("sample_size", "Units sampled", minimum=1),
            _integer("acceptance_number", "Maximum accepted defects"),
        ),
        (_out("acceptance_probability", "Probability lot is accepted"),),
        ("Simple random sampling without replacement",),
        ReviewState.REVIEWED,
        _hypergeometric_oc,
    ),
    FormulaDefinition(
        "qrm.rectifying_sampling_performance",
        COURSE_ID,
        "AOQ, ATI, and ASN for rectifying inspection",
        "AOQ=p P_a(N-n)/N; ATI=nP_a+N(1-P_a); ASN=n",
        _source(
            "Sampling_Plan_L15.pdf",
            ("2-7",),
            "Acceptance sampling — rectification performance measures",
        ),
        (
            _integer("lot_size", "Lot size", minimum=1),
            _integer("sample_size", "Single-plan sample size", minimum=1),
            _number(
                "fraction_defective",
                "Incoming fraction defective",
                minimum=0.0,
                maximum=1.0,
            ),
            _number(
                "acceptance_probability",
                "OC probability of acceptance",
                minimum=0.0,
                maximum=1.0,
            ),
        ),
        (
            _out("average_outgoing_quality", "Average outgoing fraction defective"),
            _out("average_total_inspection", "Expected units inspected", "units"),
            _out("average_sample_number", "Expected sample size", "units"),
        ),
        (
            "Rejected lots receive 100% rectifying inspection",
            "Defectives found during inspection are replaced or corrected",
            "Single-sampling plan, so ASN equals n",
        ),
        ReviewState.REVIEWED,
        _sampling_performance,
    ),
    FormulaDefinition(
        "qrm.average_run_length",
        COURSE_ID,
        "Geometric average run length",
        "ARL=1/p_signal",
        _source(
            "Quality_Reliability_Maintenance_Control_Chart_2_Note.pdf",
            ("2-7",),
            "Statistical process control — run length",
        ),
        (
            _number(
                "signal_probability",
                "Independent probability of a chart signal per sample",
                minimum=0.0,
                maximum=1.0,
                minimum_exclusive=True,
            ),
        ),
        (_out("average_run_length", "Expected samples until signal", "samples"),),
        ("Signal events are independent with constant probability",),
        ReviewState.REVIEWED,
        _arl,
    ),
    FormulaDefinition(
        "qrm.xbar_control_limits",
        COURSE_ID,
        "X-bar chart limits with known process sigma",
        "CL=mu; UCL/LCL=mu +/- z sigma/sqrt(n)",
        _source(
            "Quality_Reliability_Maintenance_Control_Chart_2_Note.pdf",
            ("33-36",),
            "Control charts — X-bar chart with known sigma",
        ),
        (
            _number("process_mean", "Target process mean", "observation"),
            _number(
                "process_standard_deviation",
                "Known process standard deviation",
                "observation",
                minimum=0.0,
                minimum_exclusive=True,
            ),
            _integer("sample_size", "Subgroup size", minimum=1),
            _number(
                "sigma_multiplier",
                "Control-limit width in standard errors",
                minimum=0.0,
                minimum_exclusive=True,
            ),
        ),
        (
            _out("lower_control_limit", "Lower control limit", "observation"),
            _out("center_line", "Chart center line", "observation"),
            _out("upper_control_limit", "Upper control limit", "observation"),
        ),
        ("Independent observations", "Stable process with known sigma"),
        ReviewState.REVIEWED,
        _xbar_limits,
    ),
    FormulaDefinition(
        "qrm.p_control_limits",
        COURSE_ID,
        "Constant-sample-size p-chart limits",
        "CL=pbar; UCL/LCL=pbar +/- z sqrt(pbar(1-pbar)/n)",
        _source(
            "Quality_Reliability_Maintenance_Control_Chart_3_Note.pdf",
            ("16-24",),
            "Control charts — p chart",
        ),
        (
            _number(
                "average_fraction_nonconforming",
                "Historical average fraction nonconforming",
                minimum=0.0,
                maximum=1.0,
            ),
            _integer("sample_size", "Constant subgroup size", minimum=1),
            _number(
                "sigma_multiplier",
                "Control-limit width in standard deviations",
                minimum=0.0,
                minimum_exclusive=True,
            ),
        ),
        (
            _out("raw_lower_control_limit", "Unclipped lower limit"),
            _out("lower_control_limit", "Practical lower limit clipped to zero"),
            _out("center_line", "Chart center line"),
            _out("upper_control_limit", "Practical upper limit clipped to one"),
            _out("raw_upper_control_limit", "Unclipped upper limit"),
        ),
        (
            "Binomial count model",
            "Constant subgroup size",
            "Normal approximation is adequate",
        ),
        ReviewState.REVIEWED,
        _p_chart_limits,
        discrepancy_warning=(
            "The source formula gives unbounded three-sigma limits. Practical "
            "proportion-chart limits are also returned clipped to [0,1]."
        ),
    ),
    FormulaDefinition(
        "qrm.exponential_reliability",
        COURSE_ID,
        "Exponential reliability distribution",
        "R(t)=exp(-lambda t); F(t)=1-R(t); MTTF=1/lambda",
        _source(
            "Quality_Reliability_Maintenance_ME3327E.pdf",
            ("9-11",),
            "Reliability distributions — exponential model",
        ),
        (
            _number(
                "time",
                "Mission time",
                "time",
                minimum=0.0,
            ),
            _number(
                "failure_rate_per_time",
                "Constant hazard rate",
                "1/time",
                minimum=0.0,
                minimum_exclusive=True,
            ),
        ),
        (
            _out("reliability", "Survival probability"),
            _out("failure_probability", "Cumulative failure probability"),
            _out("hazard_rate", "Instantaneous hazard", "1/time"),
            _out("mean_time_to_failure", "Mean lifetime", "time"),
        ),
        ("Constant failure rate", "Non-repairable item"),
        ReviewState.REVIEWED,
        _exponential_reliability,
    ),
    FormulaDefinition(
        "qrm.weibull_reliability",
        COURSE_ID,
        "Two-parameter Weibull reliability distribution",
        "R(t)=exp(-(t/eta)^beta); h(t)=beta/eta(t/eta)^(beta-1)",
        _source(
            "Quality_Reliability_Maintenance_L8.pdf",
            ("2-6",),
            "Reliability distributions — Weibull model",
        ),
        (
            _number(
                "time",
                "Mission time",
                "time",
                minimum=0.0,
                minimum_exclusive=True,
            ),
            _number(
                "scale_time",
                "Characteristic life eta",
                "time",
                minimum=0.0,
                minimum_exclusive=True,
            ),
            _number(
                "shape",
                "Weibull shape parameter beta",
                minimum=0.0,
                minimum_exclusive=True,
            ),
        ),
        (
            _out("reliability", "Survival probability"),
            _out("failure_probability", "Cumulative failure probability"),
            _out("hazard_rate", "Instantaneous hazard", "1/time"),
            _out("mean_time_to_failure", "Mean lifetime", "time"),
        ),
        ("Two-parameter Weibull model with zero location", "Non-repairable item"),
        ReviewState.REVIEWED,
        _weibull_reliability,
    ),
)


def register_formulas(registry: FormulaRegistry) -> None:
    for formula in FORMULAS:
        registry.register(formula)
