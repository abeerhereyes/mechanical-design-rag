"""Reviewed elementary-flow and aerodynamic-force formulas."""
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

COURSE_ID = "aerodynamics"
_SOURCE_DOCUMENT = "Aerodynamic_M-1.pdf"


def _source(pages: Tuple[str, ...], section: str) -> SourceReference:
    return SourceReference(_SOURCE_DOCUMENT, pages, section)


def _num(
    name: str,
    description: str,
    unit: str = "1",
    minimum=None,
    exclusive: bool = False,
) -> VariableDefinition:
    return VariableDefinition(
        name,
        description,
        unit=unit,
        minimum=minimum,
        minimum_exclusive=exclusive,
    )


def _out(name: str, description: str, unit: str = "1") -> OutputDefinition:
    return OutputDefinition(name, description, unit)


def _force_decomposition(
    normal_force_n: float, axial_force_n: float, angle_of_attack_rad: float
) -> Dict[str, float]:
    cosine = math.cos(angle_of_attack_rad)
    sine = math.sin(angle_of_attack_rad)
    return {
        "lift": normal_force_n * cosine - axial_force_n * sine,
        "drag": normal_force_n * sine + axial_force_n * cosine,
    }


def _nonzero_radius(x_m: float, y_m: float) -> float:
    radius_squared = x_m * x_m + y_m * y_m
    if radius_squared == 0:
        raise FormulaValidationError("Flow singularity is undefined at the origin")
    return radius_squared


def _source_flow(x_m: float, y_m: float, strength_m2_per_s: float) -> Dict[str, float]:
    radius_squared = _nonzero_radius(x_m, y_m)
    factor = strength_m2_per_s / (2.0 * math.pi * radius_squared)
    return {"velocity_x": factor * x_m, "velocity_y": factor * y_m}


def _doublet_flow(
    x_m: float, y_m: float, strength_m3_per_s: float
) -> Dict[str, float]:
    radius_squared = _nonzero_radius(x_m, y_m)
    denominator = 2.0 * math.pi * radius_squared * radius_squared
    return {
        "velocity_x": strength_m3_per_s * (y_m * y_m - x_m * x_m) / denominator,
        "velocity_y": -strength_m3_per_s * 2.0 * x_m * y_m / denominator,
    }


def _vortex_flow(
    x_m: float, y_m: float, circulation_m2_per_s: float
) -> Dict[str, float]:
    radius_squared = _nonzero_radius(x_m, y_m)
    factor = circulation_m2_per_s / (2.0 * math.pi * radius_squared)
    return {"velocity_x": -factor * y_m, "velocity_y": factor * x_m}


def _cylinder_flow(
    freestream_speed_m_per_s: float,
    cylinder_radius_m: float,
    radial_position_m: float,
    theta_rad: float,
    circulation_m2_per_s: float,
) -> Dict[str, float]:
    if radial_position_m < cylinder_radius_m:
        raise FormulaValidationError(
            "Radial position must be on or outside the cylinder"
        )
    ratio = (cylinder_radius_m / radial_position_m) ** 2
    radial = freestream_speed_m_per_s * (1.0 - ratio) * math.cos(theta_rad)
    tangential = (
        -freestream_speed_m_per_s * (1.0 + ratio) * math.sin(theta_rad)
        + circulation_m2_per_s / (2.0 * math.pi * radial_position_m)
    )
    return {"radial_velocity": radial, "tangential_velocity": tangential}


def _pressure_coefficient(
    local_speed_m_per_s: float, freestream_speed_m_per_s: float
) -> Dict[str, float]:
    return {
        "pressure_coefficient": 1.0
        - (local_speed_m_per_s / freestream_speed_m_per_s) ** 2
    }


def _lifting_cylinder(
    density_kg_per_m3: float,
    freestream_speed_m_per_s: float,
    circulation_m2_per_s: float,
) -> Dict[str, float]:
    return {
        "lift_per_span": (
            -density_kg_per_m3
            * freestream_speed_m_per_s
            * circulation_m2_per_s
        ),
        "drag_per_span": 0.0,
    }


_FLOW_ASSUMPTIONS = (
    "Two-dimensional, incompressible, inviscid, irrotational flow away from singularities",
    "Coordinates are measured from the singularity at the origin",
)

FORMULAS: Tuple[FormulaDefinition, ...] = (
    FormulaDefinition(
        "aero.force_decomposition",
        COURSE_ID,
        "Normal/axial force decomposition",
        "L=N cos(alpha)-A sin(alpha); D=N sin(alpha)+A cos(alpha)",
        _source(("31", "33-35"), "Aerodynamic forces"),
        (
            _num("normal_force_n", "Force normal to chord", "N"),
            _num("axial_force_n", "Force along positive chord direction", "N"),
            _num("angle_of_attack_rad", "Chord angle above freestream", "rad"),
        ),
        (_out("lift", "Lift, positive upward", "N"), _out("drag", "Drag", "N")),
        (
            "Positive normal force points above the chord",
            "Positive axial force points toward the trailing edge",
            "Positive angle of attack rotates chord counter-clockwise from freestream",
        ),
        ReviewState.REVIEWED,
        _force_decomposition,
    ),
    FormulaDefinition(
        "aero.point_source_velocity",
        COURSE_ID,
        "Point source/sink velocity",
        "(u,v)=Q(x,y)/(2 pi r^2)",
        _source(("38-40",), "Elementary source and sink flow"),
        (
            _num("x_m", "Horizontal coordinate", "m"),
            _num("y_m", "Vertical coordinate", "m"),
            _num("strength_m2_per_s", "Source strength; negative is a sink", "m^2/s"),
        ),
        (
            _out("velocity_x", "Horizontal velocity", "m/s"),
            _out("velocity_y", "Vertical velocity", "m/s"),
        ),
        _FLOW_ASSUMPTIONS + ("Positive strength denotes an outward source",),
        ReviewState.REVIEWED,
        _source_flow,
    ),
    FormulaDefinition(
        "aero.doublet_velocity",
        COURSE_ID,
        "Horizontal doublet velocity",
        "u=mu(y^2-x^2)/(2 pi r^4); v=-mu(2xy)/(2 pi r^4)",
        _source(("44-49",), "Elementary doublet flow"),
        (
            _num("x_m", "Horizontal coordinate", "m"),
            _num("y_m", "Vertical coordinate", "m"),
            _num("strength_m3_per_s", "Doublet strength", "m^3/s"),
        ),
        (
            _out("velocity_x", "Horizontal velocity", "m/s"),
            _out("velocity_y", "Vertical velocity", "m/s"),
        ),
        _FLOW_ASSUMPTIONS
        + ("Potential convention phi=mu*x/(2*pi*r^2) is used",),
        ReviewState.REVIEWED,
        _doublet_flow,
        discrepancy_warning=(
            "Doublet-strength signs and 2*pi scaling vary by text; this registry "
            "uses phi=mu*cos(theta)/(2*pi*r)."
        ),
    ),
    FormulaDefinition(
        "aero.point_vortex_velocity",
        COURSE_ID,
        "Point vortex velocity",
        "(u,v)=Gamma(-y,x)/(2 pi r^2)",
        _source(("55-57",), "Elementary vortex flow"),
        (
            _num("x_m", "Horizontal coordinate", "m"),
            _num("y_m", "Vertical coordinate", "m"),
            _num(
                "circulation_m2_per_s",
                "Circulation, positive counter-clockwise",
                "m^2/s",
            ),
        ),
        (
            _out("velocity_x", "Horizontal velocity", "m/s"),
            _out("velocity_y", "Vertical velocity", "m/s"),
        ),
        _FLOW_ASSUMPTIONS + ("Positive circulation is counter-clockwise",),
        ReviewState.REVIEWED,
        _vortex_flow,
        discrepancy_warning=(
            "Some aerodynamic texts define positive circulation clockwise. "
            "This registry defines positive circulation counter-clockwise."
        ),
    ),
    FormulaDefinition(
        "aero.cylinder_flow_velocity",
        COURSE_ID,
        "Potential flow around a circular cylinder",
        "V_r=U(1-a^2/r^2)cos(theta); "
        "V_theta=-U(1+a^2/r^2)sin(theta)+Gamma/(2 pi r)",
        _source(("50-52", "57-60"), "Uniform flow plus doublet and vortex"),
        (
            _num(
                "freestream_speed_m_per_s",
                "Freestream speed",
                "m/s",
                minimum=0.0,
                exclusive=True,
            ),
            _num(
                "cylinder_radius_m",
                "Cylinder radius",
                "m",
                minimum=0.0,
                exclusive=True,
            ),
            _num(
                "radial_position_m",
                "Distance from cylinder centre",
                "m",
                minimum=0.0,
                exclusive=True,
            ),
            _num("theta_rad", "Polar angle from positive x-axis", "rad"),
            VariableDefinition(
                "circulation_m2_per_s",
                "Circulation, positive counter-clockwise",
                unit="m^2/s",
                required=False,
                default=0.0,
            ),
        ),
        (
            _out("radial_velocity", "Radial velocity", "m/s"),
            _out("tangential_velocity", "Tangential velocity", "m/s"),
        ),
        _FLOW_ASSUMPTIONS
        + (
            "Cylinder is centred at origin",
            "Positive circulation is counter-clockwise",
        ),
        ReviewState.REVIEWED,
        _cylinder_flow,
        discrepancy_warning=(
            "Circulation sign conventions differ; positive Gamma here adds "
            "positive (counter-clockwise) tangential velocity."
        ),
    ),
    FormulaDefinition(
        "aero.pressure_coefficient",
        COURSE_ID,
        "Incompressible pressure coefficient from speed",
        "C_p=1-(V/U_inf)^2",
        _source(("53-54", "61"), "Bernoulli equation and pressure coefficient"),
        (
            _num(
                "local_speed_m_per_s",
                "Local flow speed magnitude",
                "m/s",
                minimum=0.0,
            ),
            _num(
                "freestream_speed_m_per_s",
                "Freestream speed",
                "m/s",
                minimum=0.0,
                exclusive=True,
            ),
        ),
        (_out("pressure_coefficient", "Pressure coefficient"),),
        (
            "Steady incompressible inviscid flow",
            "Local and freestream points share a Bernoulli constant",
        ),
        ReviewState.REVIEWED,
        _pressure_coefficient,
    ),
    FormulaDefinition(
        "aero.kutta_joukowski_cylinder",
        COURSE_ID,
        "Kutta–Joukowski lift for a lifting cylinder",
        "L'=-rho U_inf Gamma; D'=0",
        _source(("61-66",), "Lifting cylinder and Kutta–Joukowski theorem"),
        (
            _num(
                "density_kg_per_m3",
                "Fluid density",
                "kg/m^3",
                minimum=0.0,
                exclusive=True,
            ),
            _num(
                "freestream_speed_m_per_s",
                "Freestream speed in positive x direction",
                "m/s",
                minimum=0.0,
                exclusive=True,
            ),
            _num(
                "circulation_m2_per_s",
                "Circulation, positive counter-clockwise",
                "m^2/s",
            ),
        ),
        (
            _out("lift_per_span", "Lift per unit span, positive upward", "N/m"),
            _out("drag_per_span", "Inviscid drag per unit span", "N/m"),
        ),
        (
            "Two-dimensional steady incompressible inviscid flow",
            "Freestream is in positive x; lift is positive y",
            "Positive circulation is counter-clockwise, hence L'=-rho U Gamma",
            "Zero drag is the ideal-flow d'Alembert result",
        ),
        ReviewState.REVIEWED,
        _lifting_cylinder,
        discrepancy_warning=(
            "Many texts use clockwise-positive circulation and write "
            "L'=rho*U*Gamma. With this registry's counter-clockwise-positive "
            "convention, upward lift requires negative Gamma."
        ),
    ),
)


def register_formulas(registry: FormulaRegistry) -> None:
    for formula in FORMULAS:
        registry.register(formula)
