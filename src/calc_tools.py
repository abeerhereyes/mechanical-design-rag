"""
Phase 4 support: Real calculation tools (not LLM-generated numbers).

WHY THIS MATTERS FOR THE AGENT DESIGN (interview framing):
An LLM should never be trusted to do arithmetic/engineering calculations
itself -- it can hallucinate plausible-looking wrong numbers. These are
deterministic Python functions the agent calls as *tools*; the LLM's job is
only to (a) recognize a calculation is being asked for, (b) extract the
right numbers from the user's question, (c) call the right function, and
(d) explain the result -- never to compute the number itself. This
retrieval-vs-computation split is the core argument for having an agent
layer at all instead of pure RAG.
"""
import math


def _positive(**values: float) -> None:
    invalid = [name for name, value in values.items() if value <= 0]
    if invalid:
        raise ValueError(f"Values must be positive: {', '.join(invalid)}")


def tensile_stress_area_metric(dp_mm: float, pitch_mm: float) -> float:
    """At = (pi/4) * (dp - 0.9382*p)^2   [Section B2]. Returns mm^2."""
    _positive(dp_mm=dp_mm, pitch_mm=pitch_mm)
    if dp_mm <= 0.9382 * pitch_mm:
        raise ValueError("Major diameter must exceed 0.9382 times the pitch")
    return (math.pi / 4) * (dp_mm - 0.9382 * pitch_mm) ** 2


def bolt_preload(at_mm2: float, proof_strength_mpa: float, reused: bool = True) -> dict:
    """Fi = 0.75*Fp (reused) or 0.90*Fp (permanent), Fp = At*Sp.  [Section B1]"""
    _positive(at_mm2=at_mm2, proof_strength_mpa=proof_strength_mpa)
    fp = at_mm2 * proof_strength_mpa
    factor = 0.75 if reused else 0.90
    fi = factor * fp
    return {"proof_load_N": fp, "preload_N": fi, "factor_used": factor}


def bolt_joint_load(fi_n: float, kb: float, km: float, external_load_n: float) -> dict:
    """C = kb/(kb+km); Fb = Fi + C*P; Fm = Fi - (1-C)*P.  [Section B3]"""
    _positive(fi_n=fi_n, kb=kb, km=km)
    if external_load_n < 0:
        raise ValueError("External load cannot be negative")
    c = kb / (kb + km)
    fb = fi_n + c * external_load_n
    fm = fi_n - (1 - c) * external_load_n
    return {"C": c, "bolt_load_N": fb, "member_load_N": fm, "joint_separated": fm <= 0}


def weld_primary_shear(force_n: float, leg_size_mm: float, weld_length_mm: float) -> float:
    """tau' = F / (0.707*h*L)  [Section W2]. Returns MPa (N/mm^2)."""
    _positive(force_n=force_n, leg_size_mm=leg_size_mm, weld_length_mm=weld_length_mm)
    throat_area = 0.707 * leg_size_mm * weld_length_mm
    return force_n / throat_area


def spring_rate(g_mpa: float, d_wire_mm: float, d_mean_mm: float, n_active: float) -> float:
    """k = (G*d^4) / (8*D^3*Na)  [Section S1]. Returns N/mm."""
    _positive(g_mpa=g_mpa, d_wire_mm=d_wire_mm, d_mean_mm=d_mean_mm, n_active=n_active)
    return (g_mpa * d_wire_mm**4) / (8 * d_mean_mm**3 * n_active)


def wahl_factor(spring_index_c: float) -> float:
    """Kw = (4C-1)/(4C-4) + 0.615/C  [Section S2]."""
    if spring_index_c <= 1:
        raise ValueError("Spring index C must be > 1")
    return (4 * spring_index_c - 1) / (4 * spring_index_c - 4) + 0.615 / spring_index_c


def spring_max_shear_stress(force_n: float, d_mean_mm: float, d_wire_mm: float) -> dict:
    """tau_max = Kw * 8*F*D / (pi*d^3)  [Section S2]. Returns MPa."""
    _positive(force_n=force_n, d_mean_mm=d_mean_mm, d_wire_mm=d_wire_mm)
    c = d_mean_mm / d_wire_mm
    kw = wahl_factor(c)
    tau_uncorrected = 8 * force_n * d_mean_mm / (math.pi * d_wire_mm**3)
    return {"spring_index_C": c, "wahl_factor_Kw": kw,
            "tau_uncorrected_MPa": tau_uncorrected, "tau_max_MPa": kw * tau_uncorrected}


def bearing_l10_life_hours(c_rating_n: float, p_equiv_n: float, speed_rpm: float,
                            bearing_type: str = "ball") -> dict:
    """L10 (millions rev) = (C/P)^a, a=3 ball / 10/3 roller. Convert to hours.  [Sections BR1, BR4]"""
    _positive(c_rating_n=c_rating_n, p_equiv_n=p_equiv_n, speed_rpm=speed_rpm)
    if bearing_type not in {"ball", "roller"}:
        raise ValueError("bearing_type must be 'ball' or 'roller'")
    a = 3.0 if bearing_type == "ball" else 10.0 / 3.0
    l10_million_rev = (c_rating_n / p_equiv_n) ** a
    l10_hours = (l10_million_rev * 1e6) / (60 * speed_rpm)
    return {"exponent_a": a, "L10_million_rev": l10_million_rev, "L10_hours": l10_hours}


TOOL_REGISTRY = {
    "tensile_stress_area_metric": tensile_stress_area_metric,
    "bolt_preload": bolt_preload,
    "bolt_joint_load": bolt_joint_load,
    "weld_primary_shear": weld_primary_shear,
    "spring_rate": spring_rate,
    "wahl_factor": wahl_factor,
    "spring_max_shear_stress": spring_max_shear_stress,
    "bearing_l10_life_hours": bearing_l10_life_hours,
}

if __name__ == "__main__":
    at = tensile_stress_area_metric(dp_mm=10, pitch_mm=1.5)
    print("At (M10x1.5):", round(at, 2), "mm^2")
    print("Preload (reused, Grade 8-equiv 827 MPa):",
          bolt_preload(at, 827, reused=True))
    print("Spring rate:", round(spring_rate(g_mpa=79300, d_wire_mm=3, d_mean_mm=25, n_active=8), 2), "N/mm")
    print("Spring stress:", spring_max_shear_stress(force_n=200, d_mean_mm=25, d_wire_mm=3))
    print("Bearing life:", bearing_l10_life_hours(c_rating_n=15000, p_equiv_n=3000, speed_rpm=1800))
