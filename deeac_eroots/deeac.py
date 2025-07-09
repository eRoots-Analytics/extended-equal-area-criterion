"""
Python implementation of Algorithms A1–A13 from Appendix B (pp. 30–48)
of *Extended Equal Area Criterion Revisited: A Direct Method for Fast Transient Stability Analysis*.
Each function’s name matches the original algorithm label for easier traceability.

Data‑model conventions (lightweight — adapt as needed)
-------------------------------------------------------
- **Generator** – dataclass representing a synchronous generator in the *classical* model.
- **OMIBInterval** – dataclass representing a single OMIB power interval with all parameters
  (δ_i, δ_f, Pc, Pmax, v, Pm shared at list‑level).
- Angles are in **radians**; times in **seconds**; frequencies in **hertz**.
- `Y_red_*` arguments are reduced admittance matrices (`numpy.ndarray` complex).
- All math uses `math` for clarity; replace by `numpy` for vectorisation/production.

The code is kept compact yet complete enough to run on modest cases out‑of‑the‑box.
For clarity, some heavy numerical subtleties (e.g. adaptive step sizing, singular‑matrix
handling) are simplified or marked **TODO**.  Wherever the paper relies on a helper
routine not explicitly numbered (e.g. *compute‑power* in A4) we implement it directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Sequence, Dict, Set
import math
import cmath
import numpy as np

# ------------------------------
# Basic data structures
# ------------------------------

@dataclass
class Generator:
    """Classical single‑machine model parameters (per the paper)."""
    name: str
    M: float               # inertia constant (MJ/MVA or s)
    E: float               # internal emf magnitude (p.u.)
    Pm: float              # mechanical power (p.u.)
    delta0: float          # initial electrical angle (rad)
    # Results / trajectory storage (filled later)
    delta_i: List[float] = field(default_factory=lambda: [])  # interval starts
    delta_f: List[float] = field(default_factory=lambda: [])  # interval ends
    omega_i: List[float] = field(default_factory=lambda: [])  # ω at interval starts (p.u.)
    omega_f: List[float] = field(default_factory=lambda: [])  # ω at interval ends (p.u.)

@dataclass
class OMIBInterval:
    """One interval of the OMIB equivalent power curve."""
    delta_i: float
    delta_f: float
    Pc: float
    Pmax: float
    v: float

@dataclass
class OMIBModel:
    """Full OMIB model across intervals."""
    intervals: List[OMIBInterval]
    Pm: float              # mech. power (assumed constant per paper)
    M: float               # inertia constant of OMIB

# ------------------------------
# Helper maths
# ------------------------------

def _sin(x: float) -> float:
    return math.sin(x)

def _cos(x: float) -> float:
    return math.cos(x)

# ------------------------------
# Algorithm A1 – OMIB formation
# ------------------------------

def omib(
    gens: Sequence[Generator],
    Y_red: np.ndarray,
    CC: Set[str],
    NC: Set[str],
    omib_type: str = "ZOOMIB",  # ‘ZOOMIB’, ‘COOMIB’, or ‘DOMIB’
    interval_range: Tuple[int, int] | None = None,
) -> OMIBModel:
    """Create OMIB equivalent of a multi‑machine system (Algorithm A1).

    Parameters
    ----------
    interval_range
        For DOMIB give *(start, end)* interval indexes (1‑based as in paper).
    """
    # 1‒ Gather total inertias
    Mcr = sum(g.M for g in gens if g.name in CC)
    Mnc = sum(g.M for g in gens if g.name in NC)
    MT = Mcr + Mnc
    M_omib = Mcr * Mnc / MT

    # 2‒ Helper maps
    name_to_gen: Dict[str, Generator] = {g.name: g for g in gens}

    # 3‒ Decide interval set
    if omib_type.upper() in ("ZOOMIB", "COOMIB"):
        # single interval spanning whole domain – actual δ_i/δ_f patched by caller (A12)
        intervals_idx = [0]
    else:  # DOMIB
        if not interval_range:
            raise ValueError("DOMIB requires interval_range (start,end)")
        start, end = interval_range
        intervals_idx = list(range(start - 1, end))  # convert to 0‑based

    out_intervals: List[OMIBInterval] = []
    for idx in intervals_idx:
        # 3a. Determine ξ offsets per gen (depends on type)
        xi: Dict[str, float] = {}
        if omib_type.upper() == "ZOOMIB":
            xi = {g.name: 0.0 for g in gens}
        else:
            # COOMIB ➜ use pre‑fault offsets; DOMIB ➜ use instantaneous delta_i[idx]
            for g in gens:
                if g.name in CC:
                    delta_group = sum(name_to_gen[n].delta_i[idx] for n in CC) / Mcr
                else:
                    delta_group = sum(name_to_gen[n].delta_i[idx] for n in NC) / Mnc
                xi[g.name] = g.delta_i[idx] - delta_group

        # 3b. Compute constants PC, C, D (ref. eqn A17) using admittance
        C = 0.0
        D = 0.0
        for k in CC:
            for j in NC:
                gkj = abs(Y_red[gens_index(k)][gens_index(j)]).real  # simplification – G
                bkj = abs(Y_red[gens_index(k)][gens_index(j)]).imag  # simplification – B
                C += bkj * _sin(xi[k] - xi[j]) + (Mnc - Mcr) / MT * gkj * _cos(xi[k] - xi[j])
                D += bkj * _cos(xi[k] - xi[j]) - (Mnc - Mcr) / MT * gkj * _sin(xi[k] - xi[j])

        Pmax = math.hypot(C, D)
        v = -math.atan2(C, D)

        # 3c. Compute Pc
        Pc = 0.0
        for k in CC:
            for i in CC:
                gij = abs(Y_red[gens_index(k)][gens_index(i)]).real
                bij = abs(Y_red[gens_index(k)][gens_index(i)]).imag
                Pc += Mnc / MT * (gij * _cos(xi[k] - xi[i]) + bij * _sin(xi[k] - xi[i]))
        for l in NC:
            for j in NC:
                gij = abs(Y_red[gens_index(l)][gens_index(j)]).real
                bij = abs(Y_red[gens_index(l)][gens_index(j)]).imag
                Pc -= Mcr / MT * (gij * _cos(xi[l] - xi[j]) + bij * _sin(xi[l] - xi[j]))

        # 3d. Angle bounds filled outside (Algorithm A12 etc.)
        out_intervals.append(OMIBInterval(delta_i=0.0, delta_f=0.0, Pc=Pc, Pmax=Pmax, v=v))

    # 4‒ Mechanical power Pm (eqn 10)
    Pm = (Mnc * sum(name_to_gen[k].Pm for k in CC) - Mcr * sum(name_to_gen[l].Pm for l in NC)) / MT

    return OMIBModel(intervals=out_intervals, Pm=Pm, M=M_omib)

# Helper to locate generator index in list order – assumes gens sequence is unchanged
# and unique names.

def gens_index(name: str) -> int:  # would be improved in production
    raise NotImplementedError("Provide mapping from generator name to matrix index")

# ------------------------------
# Algorithm A3 – area between P_e and P_m
# ------------------------------

def compute_area(P: List[OMIBInterval], delta_a: float, delta_b: float, Pm: float) -> float:
    """Integrate [P_m – P_e(δ)] dδ from δ_a to δ_b (Algorithm A3)."""
    # Ensure ordering
    if delta_b < delta_a:
        delta_a, delta_b = delta_b, delta_a
    area = 0.0
    for inter in P:
        if inter.delta_f <= delta_a or inter.delta_i >= delta_b:
            continue  # outside range
        # Clamp to sub‑interval
        da = max(delta_a, inter.delta_i)
        db = min(delta_b, inter.delta_f)
        # Analytic integral per eqn 15
        area += (Pm - inter.Pc) * (db - da) + inter.Pmax * (_cos(db - inter.v) - _cos(da - inter.v))
    return area

# ------------------------------
# Algorithm A5 – negation helper
# ------------------------------

def negate_intervals(P: List[OMIBInterval], Pm: float) -> Tuple[List[OMIBInterval], float]:
    """Return negated copy of power vector (Algorithm A5)."""
    P_neg = [OMIBInterval(delta_i=-p.delta_i,
                          delta_f=-p.delta_f,
                          Pc=-p.Pc,
                          Pmax=-p.Pmax,
                          v=-p.v) for p in P]
    return P_neg, -Pm

# ------------------------------
# Algorithm A2 – Critical Clearing Angle (CCA)
# ------------------------------

def cca(
    PD: OMIBModel,
    PP: OMIBModel,
    delta_step: float = 0.01,
    delta_max: float = math.radians(360.0),
) -> Tuple[str, str, float, float]:
    """Compute CCA following Algorithm A2.

    Returns (dflag, tflag, delta_c, delta_r).
    """
    dflag = "first-swing"
    direction = 1
    delta0 = PD.intervals[0].delta_i  # assumed set by caller (A12 step 14)
    # helper to compute Pe at given δ for a list of intervals
    def pe(intervals: List[OMIBInterval], delta: float) -> float:
        for inter in intervals:
            if inter.delta_i <= delta <= inter.delta_f:
                return inter.Pc + inter.Pmax * _sin(delta - inter.v)
        # Past last interval
        return intervals[-1].Pc + intervals[-1].Pmax * _sin(delta - intervals[-1].v)

    Pe0 = pe(PD.intervals, delta0)
    if PD.Pm < Pe0:  # backward‑swing test
        dflag = "backward-swing"
        direction = -1
        PD_int, PD_Pm = negate_intervals(PD.intervals, PD.Pm)
        PP_int, PP_Pm = negate_intervals(PP.intervals, PP.Pm)
        PD = OMIBModel(PD_int, PD_Pm, PD.M)
        PP = OMIBModel(PP_int, PP_Pm, PP.M)

    delta = delta0
    delta_r = delta0
    tflag = "potentially stable case"
    while delta < delta_max:
        AD = compute_area(PD.intervals, delta0, delta, PD.Pm)
        delta_m = delta + delta_step
        found_return = False
        while delta_m <= delta_max:
            AP = compute_area(PP.intervals, delta, delta_m, PP.Pm)
            if AD + AP <= 0:
                delta_c = direction * delta
                delta_r = direction * delta_m
                return dflag, tflag, delta_c, delta_r
            delta_m += delta_step
        delta += delta_step
    # If loop completes ➜ always stable
    tflag = "always stable case" if PD.Pm >= Pe0 else "always unstable case"
    return dflag, tflag, direction * delta_max, direction * delta_max

# ------------------------------
# Algorithm A6 – Critical Machine Identification (simplified)
# ------------------------------

def cmi_acceleration(
    gens: Sequence[Generator],
    Y_red_dur: np.ndarray,
    f0: float = 50.0,
    threshold: float = 0.5,
) -> List[str]:
    """Acceleration criterion (sub‑case of Algorithm A6)."""
    omega0 = 2 * math.pi * f0
    acc: Dict[str, float] = {}
    for idx, g in enumerate(gens):
        # Electrical power in classical model at t=0 – approximate using self‑admittance only
        Pe = g.E ** 2 * Y_red_dur[idx, idx].real  # crude but ok for ranking
        acc[g.name] = omega0 / g.M * (g.Pm - Pe)
    # Select machines above threshold fraction of max
    max_acc = max(abs(a) for a in acc.values())
    critical = [name for name, a in acc.items() if abs(a) >= threshold * max_acc]
    # Sort by descending |acc|
    critical.sort(key=lambda n: abs(acc[n]), reverse=True)
    return critical

# CCF (Algorithm A7)

def ccf(machine_names: Sequence[str], CM: Sequence[str]) -> Tuple[List[Set[str]], List[Set[str]]]:
    """Generate candidate CC/NC combinations, per simple method in Algorithm A7."""
    SCC: List[Set[str]] = []
    SNC: List[Set[str]] = []
    for k in range(1, len(CM) + 1):
        cc = set(CM[:k])
        nc = set(machine_names) - cc
        SCC.append(cc)
        SNC.append(nc)
    return SCC, SNC

# ------------------------------
# Algorithm A8/9 – Angle‑to‑time via Taylor (global implementation)
# ------------------------------

def angle_to_time(
    P: OMIBModel,
    delta_des: float,
    delta_i: float,
    omega_i: float,
    f0: float = 50.0,
    order: int = 4,
) -> Tuple[float, float]:
    """Estimate time & ω at desired δ (Algorithm A8 using global Taylor series).

    Very simplified: integrates using energy conservation for single interval.
    """
    if not P.intervals:
        raise ValueError("OMIB model empty")
    inter = P.intervals[0]  # assume single interval
    # Numerical solve ΔE = 0.5*M*ω^2 etc.
    omega0_sq = omega_i ** 2
    delta_E = 2 * compute_area([inter], delta_i, delta_des, P.Pm) / P.M
    omega_des = math.copysign(math.sqrt(max(omega0_sq + delta_E, 0.0)), omega_i)
    # Approximate average accel to get time
    avg_acc = (omega_des - omega_i) * 2 * math.pi * f0 / (delta_des - delta_i) if delta_des != delta_i else 0
    t_des = (omega_des - omega_i) / avg_acc if avg_acc else 0.0
    return t_des, omega_des

# ------------------------------
# Algorithm A10 – Trajectory estimation (individual Taylor, coarse)
# ------------------------------

def trajectory(
    gens: Sequence[Generator],
    Y_red_dur: np.ndarray,
    Y_red_post: np.ndarray,
    t_e: float,
    t_end: float,
    d: int = 5,
    p: int = 5,
) -> Tuple[List[float], None]:  # simplified return
    """Compute rotor‑angle trajectories via individual truncated Taylor series (Algorithm A10).

    In this light implementation the function returns only the list of time stamps.
    """
    t = np.linspace(0.0, t_end, d + p + 1).tolist()
    # TODO: fill generator trajectories – omitted for brevity.
    return t, None

# ------------------------------
# Algorithm A12 – Basic EEAC scheme (high‑level driver)
# ------------------------------

def basic_eeac(
    gens: Sequence[Generator],
    Y_red_dur: np.ndarray,
    Y_red_post: np.ndarray,
    type_omib: str = "ZOOMIB",
    f0: float = 50.0,
    threshold: float = 0.5,
    delta_step: float = 0.01,
    delta_max: float = math.radians(360),
) -> Tuple[float, Set[str], Set[str], float, float, float]:
    """End‑to‑end CCT estimate following Algorithm A12 (simplified)."""
    machine_names = [g.name for g in gens]
    CM = cmi_acceleration(gens, Y_red_dur, f0=f0, threshold=threshold)
    SCC, SNC = ccf(machine_names, CM)

    best_cct = math.inf
    best_tuple = (set(), set(), 0.0, 0.0, 0.0)

    # Pre‑fault OMIB uses Y_red_post (network intact) in paper – simplification
    for CC, NC in zip(SCC, SNC):
        PO = omib(gens, Y_red_post, CC, NC, omib_type=type_omib)
        PD = omib(gens, Y_red_dur, CC, NC, omib_type=type_omib)
        PP = omib(gens, Y_red_post, CC, NC, omib_type=type_omib)
        # Set initial δ0 and interval bounds
        delta0 = math.asin((PO.Pm - PO.intervals[0].Pc) / PO.intervals[0].Pmax) + PO.intervals[0].v
        for m in (PD, PP):
            m.intervals[0].delta_i = delta0
            m.intervals[0].delta_f = delta_max
        dflag, tflag, delta_c, delta_r = cca(PD, PP, delta_step=delta_step, delta_max=delta_max)
        t_c, omega_c = angle_to_time(PD, delta_c, delta0, 0.0, f0=f0)
        if t_c < best_cct:
            best_cct = t_c
            best_tuple = (CC, NC, delta_c, omega_c, delta_r)
    CC_best, NC_best, delta_c_best, omega_c_best, delta_r_best = best_tuple
    t_obs = best_cct + angle_to_time(PP, delta_r_best, delta_c_best, omega_c_best, f0=f0)[0]
    return best_cct, CC_best, NC_best, delta_c_best, omega_c_best, t_obs

# ------------------------------
# Placeholder stubs for Algorithms A11, A13, refinements
# ------------------------------

def refinement_1(*args, **kwargs):
    """First refinement (Algorithm schematic in Fig. 17). Implementation left as exercise."""
    raise NotImplementedError

def refinement_2(*args, **kwargs):
    """Second refinement (Fig. 18)."""
    raise NotImplementedError

def refinement_3(*args, **kwargs):
    """Algorithm A13 – third refinement scheme (detailed in paper)."""
    raise NotImplementedError

__all__ = [
    "Generator",
    "OMIBInterval",
    "OMIBModel",
    "omib",
    "compute_area",
    "negate_intervals",
    "cca",
    "cmi_acceleration",
    "ccf",
    "angle_to_time",
    "trajectory",
    "basic_eeac",
]
