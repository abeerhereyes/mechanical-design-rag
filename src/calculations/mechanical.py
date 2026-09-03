"""Reviewed Mechanical Design formula definitions."""
from typing import Any, Dict, Tuple

from src import calc_tools

from .registry import (
    FormulaDefinition,
    FormulaRegistry,
    OutputDefinition,
    ReviewState,
    SourceReference,
    VariableDefinition,
)

COURSE_ID = "mechanical-design"


def _source(document: str, page: int, section: str) -> SourceReference:
    return SourceReference(document=document, pages=(str(page),), section=section)


def _v(
    name: str,
    description: str,
    unit: str = "1",
    minimum: float = 0.0,
    exclusive: bool = True,
) -> VariableDefinition:
    return VariableDefinition(
        name=name,
        description=description,
        unit=unit,
        minimum=minimum,
        minimum_exclusive=exclusive,
    )


def _out(name: str, description: str, unit: str = "1") -> OutputDefinition:
    return OutputDefinition(name=name, description=description, unit=unit)


def _stress_area(dp_mm: float, pitch_mm: float) -> Dict[str, float]:
    return {"area": calc_tools.tensile_stress_area_metric(dp_mm, pitch_mm)}


def _preload(
    area_mm2: float, proof_strength_mpa: float, reused: bool
) -> Dict[str, Any]:
    result = calc_tools.bolt_preload(area_mm2, proof_strength_mpa, reused)
    return {
        "proof_load": result["proof_load_N"],
        "preload": result["preload_N"],
        "preload_factor": result["factor_used"],
    }


def _joint_load(
    preload_n: float,
    bolt_stiffness_n_per_mm: float,
    member_stiffness_n_per_mm: float,
    external_load_n: float,
) -> Dict[str, Any]:
    result = calc_tools.bolt_joint_load(
        preload_n,
        bolt_stiffness_n_per_mm,
        member_stiffness_n_per_mm,
        external_load_n,
    )
    return {
        "load_fraction": result["C"],
        "bolt_load": result["bolt_load_N"],
        "member_clamp_load": result["member_load_N"],
        "joint_separated": result["joint_separated"],
    }


def _weld_shear(
    force_n: float, leg_size_mm: float, weld_length_mm: float
) -> Dict[str, float]:
    return {
        "primary_shear_stress": calc_tools.weld_primary_shear(
            force_n, leg_size_mm, weld_length_mm
        )
    }


def _spring_rate(
    shear_modulus_mpa: float,
    wire_diameter_mm: float,
    mean_diameter_mm: float,
    active_coils: float,
) -> Dict[str, float]:
    return {
        "spring_rate": calc_tools.spring_rate(
            shear_modulus_mpa,
            wire_diameter_mm,
            mean_diameter_mm,
            active_coils,
        )
    }


def _wahl(spring_index: float) -> Dict[str, float]:
    return {"wahl_factor": calc_tools.wahl_factor(spring_index)}


def _spring_stress(
    force_n: float, mean_diameter_mm: float, wire_diameter_mm: float
) -> Dict[str, float]:
    result = calc_tools.spring_max_shear_stress(
        force_n, mean_diameter_mm, wire_diameter_mm
    )
    return {
        "spring_index": result["spring_index_C"],
        "wahl_factor": result["wahl_factor_Kw"],
        "uncorrected_shear_stress": result["tau_uncorrected_MPa"],
        "maximum_shear_stress": result["tau_max_MPa"],
    }


def _bearing_life(
    dynamic_rating_n: float,
    equivalent_load_n: float,
    speed_rpm: float,
    bearing_type: str,
) -> Dict[str, float]:
    result = calc_tools.bearing_l10_life_hours(
        dynamic_rating_n, equivalent_load_n, speed_rpm, bearing_type
    )
    return {
        "life_exponent": result["exponent_a"],
        "l10_million_revolutions": result["L10_million_rev"],
        "l10_hours": result["L10_hours"],
    }


FORMULAS: Tuple[FormulaDefinition, ...] = (
    FormulaDefinition(
        formula_id="mechanical.metric_thread_tensile_area",
        course_id=COURSE_ID,
        title="Metric thread tensile stress area",
        equation="A_t = (pi/4)(d_p - 0.9382 p)^2",
        source=_source(
            "Machine Design Coursework Notes — Bolted Joints Module", 2, "B2"
        ),
        variables=(
            _v("dp_mm", "Basic major diameter", "mm"),
            _v("pitch_mm", "Thread pitch", "mm"),
        ),
        outputs=(_out("area", "Tensile stress area", "mm^2"),),
        assumptions=("Metric thread geometry", "Inputs use consistent millimetres"),
        review_state=ReviewState.REVIEWED,
        executor=_stress_area,
    ),
    FormulaDefinition(
        formula_id="mechanical.bolt_preload",
        course_id=COURSE_ID,
        title="Recommended bolt preload",
        equation="F_p=A_t S_p; F_i=(0.75 reused or 0.90 permanent)F_p",
        source=_source(
            "Machine Design Coursework Notes — Bolted Joints Module", 1, "B1"
        ),
        variables=(
            _v("area_mm2", "Tensile stress area", "mm^2"),
            _v("proof_strength_mpa", "Bolt proof strength", "MPa"),
            VariableDefinition(
                "reused",
                "True for a reusable/non-permanent connection",
                value_type="boolean",
                required=False,
                default=True,
            ),
        ),
        outputs=(
            _out("proof_load", "Proof load", "N"),
            _out("preload", "Recommended preload", "N"),
            _out("preload_factor", "Fraction of proof load"),
        ),
        assumptions=("Area in mm^2 and stress in MPa produce force in N",),
        review_state=ReviewState.REVIEWED,
        executor=_preload,
    ),
    FormulaDefinition(
        formula_id="mechanical.bolt_joint_load",
        course_id=COURSE_ID,
        title="Preloaded joint load sharing",
        equation="C=k_b/(k_b+k_m); F_b=F_i+CP; F_m=F_i-(1-C)P",
        source=_source(
            "Machine Design Coursework Notes — Bolted Joints Module", 3, "B3"
        ),
        variables=(
            _v("preload_n", "Initial preload", "N"),
            _v("bolt_stiffness_n_per_mm", "Bolt stiffness", "N/mm"),
            _v("member_stiffness_n_per_mm", "Member stiffness", "N/mm"),
            _v(
                "external_load_n",
                "Applied tensile load",
                "N",
                minimum=0.0,
                exclusive=False,
            ),
        ),
        outputs=(
            _out("load_fraction", "External load fraction carried by bolt"),
            _out("bolt_load", "Resultant bolt load", "N"),
            _out("member_clamp_load", "Remaining member clamp load", "N"),
            _out("joint_separated", "Whether clamp load is exhausted"),
        ),
        assumptions=("Linear elastic stiffnesses", "Load is tensile"),
        review_state=ReviewState.REVIEWED,
        executor=_joint_load,
    ),
    FormulaDefinition(
        formula_id="mechanical.weld_primary_shear",
        course_id=COURSE_ID,
        title="Fillet weld primary shear",
        equation="tau = F/(0.707 h L)",
        source=_source(
            "Machine Design Coursework Notes — Welded Joints Module", 2, "W2"
        ),
        variables=(
            _v("force_n", "Direct shear force", "N"),
            _v("leg_size_mm", "Fillet weld leg size", "mm"),
            _v("weld_length_mm", "Total weld length", "mm"),
        ),
        outputs=(_out("primary_shear_stress", "Primary shear stress", "MPa"),),
        assumptions=("Uniform direct loading over effective throat",),
        review_state=ReviewState.REVIEWED,
        executor=_weld_shear,
    ),
    FormulaDefinition(
        formula_id="mechanical.helical_spring_rate",
        course_id=COURSE_ID,
        title="Helical compression spring rate",
        equation="k=Gd^4/(8D^3N_a)",
        source=_source(
            "Machine Design Coursework Notes — Springs Module", 1, "S1"
        ),
        variables=(
            _v("shear_modulus_mpa", "Wire shear modulus", "MPa"),
            _v("wire_diameter_mm", "Wire diameter", "mm"),
            _v("mean_diameter_mm", "Mean coil diameter", "mm"),
            _v("active_coils", "Number of active coils"),
        ),
        outputs=(_out("spring_rate", "Spring rate", "N/mm"),),
        assumptions=("Round wire", "Linear elastic response", "No coil bind"),
        review_state=ReviewState.REVIEWED,
        executor=_spring_rate,
    ),
    FormulaDefinition(
        formula_id="mechanical.wahl_factor",
        course_id=COURSE_ID,
        title="Wahl spring correction factor",
        equation="K_w=(4C-1)/(4C-4)+0.615/C",
        source=_source(
            "Machine Design Coursework Notes — Springs Module", 2, "S2"
        ),
        variables=(
            VariableDefinition(
                "spring_index",
                "Mean coil diameter divided by wire diameter",
                minimum=1.0,
                minimum_exclusive=True,
            ),
        ),
        outputs=(_out("wahl_factor", "Wahl correction factor"),),
        assumptions=("Round-wire helical spring",),
        review_state=ReviewState.REVIEWED,
        executor=_wahl,
    ),
    FormulaDefinition(
        formula_id="mechanical.spring_max_shear_stress",
        course_id=COURSE_ID,
        title="Wahl-corrected spring shear stress",
        equation="tau_max=K_w 8FD/(pi d^3)",
        source=_source(
            "Machine Design Coursework Notes — Springs Module", 2, "S2"
        ),
        variables=(
            _v("force_n", "Axial spring force", "N"),
            _v("mean_diameter_mm", "Mean coil diameter", "mm"),
            _v("wire_diameter_mm", "Wire diameter", "mm"),
        ),
        outputs=(
            _out("spring_index", "Spring index"),
            _out("wahl_factor", "Wahl correction factor"),
            _out("uncorrected_shear_stress", "Nominal wire shear stress", "MPa"),
            _out("maximum_shear_stress", "Corrected maximum shear stress", "MPa"),
        ),
        assumptions=("Round-wire helical spring",),
        review_state=ReviewState.REVIEWED,
        executor=_spring_stress,
    ),
    FormulaDefinition(
        formula_id="mechanical.bearing_l10_life",
        course_id=COURSE_ID,
        title="Rolling bearing L10 life",
        equation="L_10=(C/P)^a; L_h=10^6 L_10/(60n)",
        source=_source(
            "Machine Design Coursework Notes — Bearings Module", 1, "BR1/BR4"
        ),
        variables=(
            _v("dynamic_rating_n", "Basic dynamic load rating", "N"),
            _v("equivalent_load_n", "Equivalent dynamic load", "N"),
            _v("speed_rpm", "Rotational speed", "rpm"),
            VariableDefinition(
                "bearing_type",
                "Rolling element type",
                value_type="string",
                required=False,
                default="ball",
                choices=("ball", "roller"),
            ),
        ),
        outputs=(
            _out("life_exponent", "Load-life exponent"),
            _out("l10_million_revolutions", "L10 life", "10^6 rev"),
            _out("l10_hours", "L10 life", "h"),
        ),
        assumptions=("Constant equivalent load and speed",),
        review_state=ReviewState.REVIEWED,
        executor=_bearing_life,
    ),
)


def register_formulas(registry: FormulaRegistry) -> None:
    for formula in FORMULAS:
        registry.register(formula)
