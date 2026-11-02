import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from CoolProp.CoolProp import PropsSI, PhaseSI, get_global_param_string, AbstractState
from CoolProp.HumidAirProp import HAPropsSI

st.set_page_config(page_title="ThermoTools", page_icon="🌡️", layout="wide")

# -------------------- CHEMICAL LIBRARY --------------------
FLUID_CATEGORIES = {
    "Light Alkanes": ["Methane", "Ethane", "Propane", "n-Butane", "IsoButane", "n-Pentane", "IsoPentane", "n-Hexane", "n-Heptane", "n-Octane", "n-Decane"],
    "Alkenes and Unsaturated Hydrocarbons": ["Ethylene", "Propylene", "1-Butene", "cis-2-Butene", "trans-2-Butene", "IsoButylene", "1-Hexene", "1-Octene", "Acetylene", "1,3-Butadiene"],
    "Process Gases": ["Hydrogen", "Nitrogen", "Oxygen", "CarbonDioxide", "CarbonMonoxide", "Ammonia"],
    "Water and Steam": ["Water", "Steam"],
    "Aromatics and Other Hydrocarbons": ["Benzene", "Toluene", "EthylBenzene", "Styrene", "CycloHexane"],
    "Solvents and Refrigerants": ["Methanol", "Ethanol", "Acetone", "R134a"]
}
AVAILABLE = set(get_global_param_string("FluidsList").split(","))
FLUID_CATEGORIES = {k: [f for f in v if f in AVAILABLE or f == "Steam"] for k, v in FLUID_CATEGORIES.items()}
FLUID_CATEGORIES = {k: v for k, v in FLUID_CATEGORIES.items() if v}

# -------------------- UNIT CONVERSIONS --------------------
PSI_TO_PA = 6894.757293
JKG_TO_BTULB = 0.000429922614
KG_M3_TO_LB_FT3 = 0.06242796

def f_to_k(x): return (x - 32) * 5 / 9 + 273.15
def k_to_f(x): return (x - 273.15) * 9 / 5 + 32
def p_to_si(x, u): return x * PSI_TO_PA if u == "Imperial" else x * 1e6
def p_from_si(x, u): return x / PSI_TO_PA if u == "Imperial" else x / 1e6
def t_to_si(x, u): return f_to_k(x) if u == "Imperial" else x + 273.15
def t_from_si(x, u): return k_to_f(x) if u == "Imperial" else x - 273.15
def h_from_si(x, u): return x * JKG_TO_BTULB if u == "Imperial" else x / 1000
def rho_from_si(x, u): return x * KG_M3_TO_LB_FT3 if u == "Imperial" else x
def s_from_si(x, u): return x * JKG_TO_BTULB if u == "Imperial" else x / 1000
def cp_from_si(x, u): return x * JKG_TO_BTULB * 5/9 if u == "Imperial" else x / 1000
def jt_from_si(x, u): return x * PSI_TO_PA * 9/5 if u == "Imperial" else x * 1e6

PUNIT = lambda u: "psia" if u == "Imperial" else "MPa"
TUNIT = lambda u: "°F" if u == "Imperial" else "°C"
HUNIT = lambda u: "BTU/lb" if u == "Imperial" else "kJ/kg"
PLABEL = lambda u: f"Pressure ({PUNIT(u)})"
TLABEL = lambda u: f"Temperature ({TUNIT(u)})"
HLABEL = lambda u: f"Enthalpy ({HUNIT(u)})"

# -------------------- PROPERTY FUNCTIONS --------------------
def prop(output, key1, val1, key2=None, val2=None):
    try:
        if key1 is None: return PropsSI(output, FLUID)
        return PropsSI(output, key1, val1, key2, val2, FLUID)
    except Exception:
        return None

def phase_name(p, t):
    pc, tc = PropsSI("Pcrit", FLUID), PropsSI("Tcrit", FLUID)
    # CoolProp's supercritical_gas means T>Tc but P<Pc: display as gas.
    if t > tc and p > pc:
        return "Supercritical Fluid"
    if t > tc and p <= pc:
        return "Gas"
    try:
        phase = PhaseSI("P", p, "T", t, FLUID)
    except Exception:
        return "Phase unavailable"
    names = {"liquid":"Liquid", "gas":"Gas", "twophase":"Two-phase", "supercritical":"Supercritical Fluid", "supercritical_gas":"Gas", "supercritical_liquid":"Dense Fluid"}
    return names.get(phase, phase.replace("_", " ").title())

def state_pt(p, t):
    return {
        "P": p, "T": t,
        "H": PropsSI("H", "P", p, "T", t, FLUID),
        "S": PropsSI("S", "P", p, "T", t, FLUID),
        "D": PropsSI("D", "P", p, "T", t, FLUID),
        "U": PropsSI("U", "P", p, "T", t, FLUID),
        "CP": prop("C", "P", p, "T", t),
        "CV": prop("O", "P", p, "T", t),
        "Z": prop("Z", "P", p, "T", t),
        "phase": phase_name(p, t)
    }

def jt(p, t):
    return PropsSI("d(T)/d(P)|H", "P", p, "T", t, FLUID)

# -------------------- MODEL RANGE HELPERS --------------------
@st.cache_data(show_spinner=False)
def get_pure_fluid_limits(fluid, backend="HEOS"):
    """Query limits from the selected CoolProp backend when available."""
    result = {"Tmin": None, "Tmax": None, "pmax": None, "pmin": None,
              "Tcrit": None, "pcrit": None}
    try:
        model = AbstractState(backend, fluid)
    except Exception:
        return result
    for key, method in (("Tmin", "Tmin"), ("Tmax", "Tmax"),
                        ("pmax", "pmax"), ("pmin", "pmin"),
                        ("Tcrit", "T_critical"), ("pcrit", "p_critical")):
        try:
            value = float(getattr(model, method)())
            if np.isfinite(value) and value > 0:
                result[key] = value
        except Exception:
            pass
    return result


def model_range_warnings(fluid, units, states):
    """Return messages for values outside *reported* limits, not a phase check."""
    limits = get_pure_fluid_limits(fluid.split("::")[-1], fluid.split("::")[0] if "::" in fluid else "HEOS")
    warnings = []
    for label, p, t in states:
        if t is not None:
            for key, word, test in (("Tmin", "below", lambda x, lim: x < lim),
                                    ("Tmax", "above", lambda x, lim: x > lim)):
                lim = limits[key]
                if lim is not None and test(t, lim):
                    warnings.append(f"{label}: {t_from_si(t, units):,.2f} {TUNIT(units)} is {word} "
                                    f"the model's reported {key} of {t_from_si(lim, units):,.2f} {TUNIT(units)}.")
        if p is not None:
            for key, word, test in (("pmin", "below", lambda x, lim: x < lim),
                                    ("pmax", "above", lambda x, lim: x > lim)):
                lim = limits[key]
                if lim is not None and test(p, lim):
                    warnings.append(f"{label}: {p_from_si(p, units):,.2f} {PUNIT(units)} is {word} "
                                    f"the model's reported {key} of {p_from_si(lim, units):,.2f} {PUNIT(units)}.")
    return warnings


def show_model_warnings(fluid, units, states):
    for message in model_range_warnings(fluid, units, states):
        st.warning(("Reported backend limits (not a validation guarantee): " if EOS != "HEOS" else "CoolProp model range: ") + message)


def explain_coolprop_error(exc, fluid, units):
    message = str(exc)
    lower = message.lower()
    if "melting line" in lower or "melting curve" in lower:
        st.error(f"CoolProp cannot evaluate this {fluid} state because its melting-line "
                 "correlation does not cover the requested pressure or temperature. "
                 "This is a model/calculation boundary, not proof the physical state is impossible.")
        import re
        match = re.search(r"bounds are\s*([+\-0-9.eE]+)\s*,\s*([+\-0-9.eE]+)\s*Pa", message, re.I)
        if match:
            try:
                lo, hi = map(float, match.groups())
                st.info(f"Melting-line correlation pressure bounds reported by CoolProp: "
                        f"{p_from_si(lo, units):,.2f}–{p_from_si(hi, units):,.2f} {PUNIT(units)}. "
                        "These are not necessarily the fluid EOS's overall pressure limits.")
            except (ValueError, OverflowError):
                pass
    elif any(x in lower for x in ("out of range", "outside the range", "out of bounds", "above the maximum", "below the minimum")):
        st.error(f"CoolProp reports that this {fluid} calculation is outside a supported "
                 "range. Check the sidebar model limits and the detailed error below.")
    else:
        st.error(f"CoolProp could not calculate this {fluid} state. The solver may have "
                 "encountered a phase boundary, unsupported input pair, or model limitation.")
    with st.expander("Technical details from CoolProp"):
        st.code(message)


# -------------------- SIDEBAR --------------------
st.sidebar.title("ThermoTools")
discipline = st.sidebar.radio("Discipline", ["Thermodynamics", "Mechanical Engineering"], key="discipline")
page = st.sidebar.radio("Mechanical tools", ["Cyclic & Fatigue Loading"], key="mechanical_page") if discipline == "Mechanical Engineering" else st.sidebar.radio("Navigation", ["Home", "Pressure-Enthalpy Diagram", "State Point Calculator", "Isenthalpic Expansion", "Joule-Thomson Calculator", "Mixture Builder", "Mixture P-H Diagram", "EOS Comparison", "Steam Tables (IF97)", "Psychrometrics", "Pipe Frost Growth"])
units = st.sidebar.radio("Unit System", ["Imperial", "SI"], horizontal=True)
st.sidebar.divider()
st.sidebar.subheader("Fluid Selection")
category = st.sidebar.selectbox("Chemical category", list(FLUID_CATEGORIES))
fluids = FLUID_CATEGORIES[category]
default_idx = fluids.index("Ethylene") if "Ethylene" in fluids else 0
SELECTED_CHEMICAL = st.sidebar.selectbox("Chemical", fluids, index=default_idx)
# Steam is a user-friendly alias: liquid water and steam are the same substance.
PURE_FLUID = "Water" if SELECTED_CHEMICAL == "Steam" else SELECTED_CHEMICAL
EOS_OPTIONS = ["HEOS", "IF97", "PR", "SRK"] if PURE_FLUID == "Water" else ["HEOS", "PR", "SRK"]
EOS_LABELS = {"HEOS": "Helmholtz (HEOS)", "IF97": "IAPWS Industrial Water/Steam (IF97)",
              "PR": "Peng–Robinson (PR)", "SRK": "Soave–Redlich–Kwong (SRK)"}
EOS = st.sidebar.selectbox("Equation of state", EOS_OPTIONS, format_func=lambda x: EOS_LABELS[x],
                           key="selected_eos")
FLUID = f"{EOS}::{PURE_FLUID}"
st.sidebar.caption("Mixture tools currently use HEOS regardless of this selection.")
if PURE_FLUID == "Water":
    st.sidebar.caption("Water and Steam both use the Water fluid model; phase is determined by pressure and temperature.")
if EOS == "IF97":
    st.sidebar.info("IF97 is the industrial water/steam formulation. Some properties (such as Z, certain derivatives, and critical-point enthalpy) may be unavailable through CoolProp.")
old_fluid = st.session_state.get("active_fluid")
if old_fluid is not None and old_fluid != FLUID:
    st.session_state.ph_points = []
    st.session_state.ph_paths = []
    st.sidebar.info("Fluid changed; saved state points and paths were cleared.")
st.session_state.active_fluid = FLUID
st.sidebar.caption(f"Active fluid: {SELECTED_CHEMICAL} ({FLUID})")
st.sidebar.divider()
with st.sidebar.expander("CoolProp model limits", expanded=True):
    st.caption(f"Selected backend: {EOS} / {PURE_FLUID}")
    _limits = get_pure_fluid_limits(PURE_FLUID, EOS)
    if EOS != "HEOS":
        st.info(f"Limits shown are reported by {EOS} when available, not experimentally validated operating limits. Unsupported properties or flashes may have narrower ranges.")
    for _label, _key, _convert, _unit in [
        ("Minimum temperature", "Tmin", t_from_si, TUNIT(units)),
        ("Maximum temperature", "Tmax", t_from_si, TUNIT(units)),
        ("Minimum pressure", "pmin", p_from_si, PUNIT(units)),
        ("Maximum pressure", "pmax", p_from_si, PUNIT(units)),
        ("Critical temperature", "Tcrit", t_from_si, TUNIT(units)),
        ("Critical pressure", "pcrit", p_from_si, PUNIT(units)),
    ]:
        _value = _limits[_key]
        st.write(f"**{_label}:** " + (f"{_convert(_value, units):,.2f} {_unit}" if _value is not None else "Not reported"))
    st.caption("Reported EOS limits are not a guarantee that every P–T state is calculable. "
               "Melting-line and phase-flash restrictions can be narrower. "
               "For mixtures, component limits do not define the mixture's valid range.")


# -------------------- STEAM TABLES (ALWAYS IF97) --------------------
# This page deliberately bypasses the sidebar EOS: all values use IF97::Water.
STEAM_FLUID = "IF97::Water"

@st.cache_data(show_spinner=False)
def steam_sat_table(mode, low, high, count):
    """Generate saturated liquid/vapor rows; invalid endpoints are skipped."""
    rows = []
    for value in np.linspace(low, high, count):
        try:
            key = "P" if mode == "Pressure" else "T"
            p = value if key == "P" else PropsSI("P", "T", value, "Q", 0, STEAM_FLUID)
            t = value if key == "T" else PropsSI("T", "P", value, "Q", 0, STEAM_FLUID)
            liquid = {v: PropsSI(v, key, value, "Q", 0, STEAM_FLUID) for v in ("H", "S", "D")}
            vapor = {v: PropsSI(v, key, value, "Q", 1, STEAM_FLUID) for v in ("H", "S", "D")}
            rows.append({"P": p, "T": t, "hf": liquid["H"], "hg": vapor["H"],
                         "hfg": vapor["H"] - liquid["H"], "sf": liquid["S"],
                         "sg": vapor["S"], "vf": 1/liquid["D"], "vg": 1/vapor["D"]})
        except (ValueError, OverflowError):
            continue
    return pd.DataFrame(rows)


def steam_sat_display(df, unit_system):
    if df.empty:
        return df
    is_imp = unit_system == "Imperial"
    vfactor = 16.018463 if is_imp else 1.0  # m³/kg -> ft³/lb
    return pd.DataFrame({
        f"Pressure ({PUNIT(unit_system)})": df.P.map(lambda x: p_from_si(x, unit_system)),
        f"Temperature ({TUNIT(unit_system)})": df["T"].map(lambda x: t_from_si(x, unit_system)),
        f"hf ({HUNIT(unit_system)})": df.hf.map(lambda x: h_from_si(x, unit_system)),
        f"hg ({HUNIT(unit_system)})": df.hg.map(lambda x: h_from_si(x, unit_system)),
        f"hfg ({HUNIT(unit_system)})": df.hfg.map(lambda x: h_from_si(x, unit_system)),
        f"sf ({'BTU/lb-°R' if is_imp else 'kJ/kg-K'})": df.sf.map(lambda x: s_from_si(x, unit_system)),
        f"sg ({'BTU/lb-°R' if is_imp else 'kJ/kg-K'})": df.sg.map(lambda x: s_from_si(x, unit_system)),
        f"vf ({'ft³/lb' if is_imp else 'm³/kg'})": df.vf * vfactor,
        f"vg ({'ft³/lb' if is_imp else 'm³/kg'})": df.vg * vfactor,
    })


if page == "Steam Tables (IF97)":
    st.title("Steam Tables · IAPWS-IF97")
    st.caption("Dedicated water/steam calculations. This page always uses CoolProp IF97::Water, "
               "regardless of the chemical or EOS selected in the sidebar. "
               "All pressures are absolute; values are specific (per unit mass).")
    tab_sat, tab_state, tab_super, tab_plots = st.tabs([
        "Saturation tables", "State lookup", "Superheated steam", "Property graphs"
    ])
    pc_water = 22.064e6
    tc_water = 647.096
    with tab_sat:
        st.subheader("Saturated liquid and saturated vapor")
        sat_basis = st.radio("Table indexed by", ["Pressure", "Temperature"], horizontal=True, key="steam_basis")
        col1, col2, col3 = st.columns(3)
        if sat_basis == "Pressure":
            if units == "Imperial":
                lo_default, hi_default = 1.0, 3000.0
                lo_bound, hi_bound = 0.1, 3200.0
            else:
                lo_default, hi_default = 0.01, 20.0
                lo_bound, hi_bound = 0.001, 22.0
            with col1: low = st.number_input(f"Minimum pressure ({PUNIT(units)})", min_value=lo_bound, max_value=hi_bound, value=lo_default, key="sat_min_p")
            with col2: high = st.number_input(f"Maximum pressure ({PUNIT(units)})", min_value=lo_bound, max_value=hi_bound, value=hi_default, key="sat_max_p")
            low_si, high_si = p_to_si(low, units), p_to_si(high, units)
        else:
            if units == "Imperial":
                lo_default, hi_default = 100.0, 650.0
                lo_bound, hi_bound = 32.1, 705.0
            else:
                lo_default, hi_default = 40.0, 340.0
                lo_bound, hi_bound = 0.1, 373.9
            with col1: low = st.number_input(f"Minimum temperature ({TUNIT(units)})", min_value=lo_bound, max_value=hi_bound, value=lo_default, key="sat_min_t")
            with col2: high = st.number_input(f"Maximum temperature ({TUNIT(units)})", min_value=lo_bound, max_value=hi_bound, value=hi_default, key="sat_max_t")
            low_si, high_si = t_to_si(low, units), t_to_si(high, units)
        with col3: count = st.slider("Table rows", 5, 150, 30, key="sat_count")
        if low_si >= high_si:
            st.warning("The minimum must be below the maximum.")
        elif (sat_basis == "Pressure" and high_si >= pc_water) or (sat_basis == "Temperature" and high_si >= tc_water):
            st.warning("Saturation tables require values below water's critical point (22.064 MPa, 373.946°C).")
        else:
            sat = steam_sat_table(sat_basis, low_si, high_si, count)
            if sat.empty:
                st.error("No valid saturation states were calculated for this range.")
            else:
                view = steam_sat_display(sat, units)
                st.dataframe(view, use_container_width=True, hide_index=True)
                st.download_button("Download saturation table (CSV)", view.to_csv(index=False).encode(),
                                   file_name="steam_saturation_IF97.csv", mime="text/csv")
                st.caption("f = saturated liquid; g = saturated vapor; fg = vapor minus liquid. "
                           "The liquid/vapor properties coincide at the critical point, which is excluded here.")
    with tab_state:
        st.subheader("Water/steam state lookup")
        state_mode = st.radio("Known independent properties", ["Pressure + Temperature", "Pressure + Quality", "Temperature + Quality"], horizontal=True)
        c1, c2 = st.columns(2)
        try:
            if state_mode == "Pressure + Temperature":
                with c1: p_in = st.number_input(f"Pressure ({PUNIT(units)})", min_value=0.001, value=14.696 if units == "Imperial" else 0.101325, key="steam_pt_p")
                with c2: t_in = st.number_input(f"Temperature ({TUNIT(units)})", value=400.0 if units == "Imperial" else 200.0, key="steam_pt_t")
                key1, val1, key2, val2 = "P", p_to_si(p_in, units), "T", t_to_si(t_in, units)
            elif state_mode == "Pressure + Quality":
                with c1: p_in = st.number_input(f"Pressure ({PUNIT(units)})", min_value=0.001, value=14.696 if units == "Imperial" else 0.101325, key="steam_pq_p")
                with c2: q = st.slider("Vapor quality (mass fraction)", 0.0, 1.0, 1.0, 0.01, key="steam_pq_q")
                key1, val1, key2, val2 = "P", p_to_si(p_in, units), "Q", q
            else:
                with c1: t_in = st.number_input(f"Temperature ({TUNIT(units)})", value=212.0 if units == "Imperial" else 100.0, key="steam_tq_t")
                with c2: q = st.slider("Vapor quality (mass fraction)", 0.0, 1.0, 1.0, 0.01, key="steam_tq_q")
                key1, val1, key2, val2 = "T", t_to_si(t_in, units), "Q", q
            vals = {}
            for k in ("P", "T", "H", "S", "D", "U", "C"):
                try: vals[k] = PropsSI(k, key1, val1, key2, val2, STEAM_FLUID)
                except (ValueError, OverflowError): vals[k] = None
            if vals["P"] is None or vals["T"] is None:
                st.error("IF97 could not resolve this state. Check the specified inputs and IF97 validity range.")
            else:
                if state_mode == "Pressure + Temperature":
                    try: phase = PhaseSI("P", vals["P"], "T", vals["T"], STEAM_FLUID)
                    except ValueError: phase = "Unavailable"
                else:
                    phase = "Saturated liquid" if q == 0 else "Saturated vapor" if q == 1 else "Two-phase mixture"
                st.write(f"**Phase:** {phase}")
                metrics = [
                    ("Pressure", p_from_si(vals["P"], units), PUNIT(units)),
                    ("Temperature", t_from_si(vals["T"], units), TUNIT(units)),
                    ("Enthalpy", h_from_si(vals["H"], units) if vals["H"] is not None else None, HUNIT(units)),
                    ("Entropy", s_from_si(vals["S"], units) if vals["S"] is not None else None, "BTU/lb-°R" if units == "Imperial" else "kJ/kg-K"),
                    ("Density", rho_from_si(vals["D"], units) if vals["D"] is not None else None, "lb/ft³" if units == "Imperial" else "kg/m³"),
                    ("Internal energy", h_from_si(vals["U"], units) if vals["U"] is not None else None, HUNIT(units)),
                    ("Specific volume", (1 / vals["D"]) * (16.018463 if units == "Imperial" else 1) if vals["D"] else None, "ft³/lb" if units == "Imperial" else "m³/kg"),
                    ("Cp", cp_from_si(vals["C"], units) if vals["C"] is not None else None, "BTU/lb-°F" if units == "Imperial" else "kJ/kg-K"),
                ]
                cols = st.columns(4)
                for i, (name, value, unit) in enumerate(metrics):
                    cols[i % 4].metric(name, f"{value:,.4g} {unit}" if value is not None and np.isfinite(value) else "N/A")
                if state_mode != "Pressure + Temperature":
                    st.caption("Cp is undefined in the two-phase region; N/A is expected for some states.")
        except (ValueError, OverflowError) as exc:
            st.error(f"IF97 calculation failed: {exc}")
    with tab_super:
        st.subheader("Superheated steam property table")
        st.caption("Specify pressures and temperatures; only single-phase vapor states are included.")
        ca, cb, cc = st.columns(3)
        with ca: pressure_text = st.text_input(f"Pressures ({PUNIT(units)}), comma-separated", "14.696, 100, 500" if units == "Imperial" else "0.1, 0.5, 2", key="steam_pressures")
        with cb: t_low = st.number_input(f"Minimum temperature ({TUNIT(units)})", value=250.0 if units == "Imperial" else 150.0, key="steam_super_tlow")
        with cc: t_high = st.number_input(f"Maximum temperature ({TUNIT(units)})", value=800.0 if units == "Imperial" else 450.0, key="steam_super_thigh")
        ntemps = st.slider("Temperatures per pressure", 3, 40, 12, key="steam_super_n")
        if t_low >= t_high:
            st.warning("Maximum temperature must exceed minimum temperature.")
        else:
            try:
                pressures = [float(v.strip()) for v in pressure_text.split(",") if v.strip()]
                if not pressures or len(pressures) > 12 or any(v <= 0 for v in pressures):
                    raise ValueError("Enter 1–12 positive pressures.")
                rows = []
                for pp in pressures:
                    for tt in np.linspace(t_low, t_high, ntemps):
                        p_si, t_si = p_to_si(pp, units), t_to_si(tt, units)
                        try:
                            phase = PhaseSI("P", p_si, "T", t_si, STEAM_FLUID)
                            if phase not in ("gas", "supercritical_gas"):
                                continue
                            h = PropsSI("H", "P", p_si, "T", t_si, STEAM_FLUID)
                            entropy = PropsSI("S", "P", p_si, "T", t_si, STEAM_FLUID)
                            rho = PropsSI("D", "P", p_si, "T", t_si, STEAM_FLUID)
                            rows.append({f"Pressure ({PUNIT(units)})": pp, f"Temperature ({TUNIT(units)})": tt,
                                         f"Enthalpy ({HUNIT(units)})": h_from_si(h, units),
                                         f"Entropy ({'BTU/lb-°R' if units == 'Imperial' else 'kJ/kg-K'})": s_from_si(entropy, units),
                                         f"Specific volume ({'ft³/lb' if units == 'Imperial' else 'm³/kg'})": (1/rho)*(16.018463 if units == 'Imperial' else 1)})
                        except (ValueError, OverflowError):
                            continue
                table = pd.DataFrame(rows)
                if table.empty:
                    st.warning("No valid vapor states in this grid. Increase temperature or lower pressure.")
                else:
                    st.dataframe(table, use_container_width=True, hide_index=True)
                    st.download_button("Download superheated table (CSV)", table.to_csv(index=False).encode(),
                                       file_name="steam_superheated_IF97.csv", mime="text/csv")
                    fig = go.Figure()
                    xkey = f"Temperature ({TUNIT(units)})"
                    ykey = f"Enthalpy ({HUNIT(units)})"
                    for pp in pressures:
                        part = table[table[f"Pressure ({PUNIT(units)})"] == pp]
                        if not part.empty:
                            fig.add_trace(go.Scatter(x=part[xkey], y=part[ykey], mode="lines+markers", name=f"{pp:g} {PUNIT(units)}"))
                    fig.update_layout(title="Superheated steam enthalpy along isobars", xaxis_title=xkey, yaxis_title=ykey)
                    st.plotly_chart(fig, use_container_width=True)
            except ValueError as exc:
                st.error(f"Invalid pressure list: {exc}")
    with tab_plots:
        st.subheader("Saturation property graphs")
        st.caption("Curves generated with IF97 at saturation. The near-critical endpoint is excluded.")
        graph = st.selectbox("Graph", ["Saturation pressure vs temperature", "Saturated liquid/vapor enthalpy vs temperature",
                                       "Latent heat of vaporization vs temperature", "Saturated liquid/vapor entropy vs temperature",
                                       "Saturated liquid/vapor specific volume vs temperature"])
        plot_data = steam_sat_table("Temperature", 273.16, 646.5, 180)
        if plot_data.empty:
            st.error("Unable to generate saturation curves from IF97.")
        else:
            x = plot_data["T"].map(lambda v: t_from_si(v, units))
            fig = go.Figure()
            if graph == "Saturation pressure vs temperature":
                fig.add_trace(go.Scatter(x=x, y=plot_data["P"].map(lambda v: p_from_si(v, units)), name="Saturation pressure"))
                ylabel = PLABEL(units)
                fig.update_yaxes(type="log")
            elif graph == "Saturated liquid/vapor enthalpy vs temperature":
                for key, label in (("hf", "Saturated liquid"), ("hg", "Saturated vapor")):
                    fig.add_trace(go.Scatter(x=x, y=plot_data[key].map(lambda v: h_from_si(v, units)), name=label))
                ylabel = HLABEL(units)
            elif graph == "Latent heat of vaporization vs temperature":
                fig.add_trace(go.Scatter(x=x, y=plot_data["hfg"].map(lambda v: h_from_si(v, units)), name="hfg"))
                ylabel = HLABEL(units)
            elif graph == "Saturated liquid/vapor entropy vs temperature":
                for key, label in (("sf", "Saturated liquid"), ("sg", "Saturated vapor")):
                    fig.add_trace(go.Scatter(x=x, y=plot_data[key].map(lambda v: s_from_si(v, units)), name=label))
                ylabel = "Entropy (BTU/lb-°R)" if units == "Imperial" else "Entropy (kJ/kg-K)"
            else:
                for key, label in (("vf", "Saturated liquid"), ("vg", "Saturated vapor")):
                    fig.add_trace(go.Scatter(x=x, y=plot_data[key] * (16.018463 if units == "Imperial" else 1), name=label))
                ylabel = "Specific volume (ft³/lb)" if units == "Imperial" else "Specific volume (m³/kg)"
                fig.update_yaxes(type="log")
            fig.update_layout(title=graph, xaxis_title=TLABEL(units), yaxis_title=ylabel,
                              hovermode="x unified", legend_title="IF97 saturation")
            st.plotly_chart(fig, use_container_width=True)



# -------------------- PSYCHROMETRICS (COOLPROP HUMID AIR) --------------------
# All moist-air enthalpies, humidity ratios and specific volumes are per kg of DRY air.
# Psychrometrics is independent of the pure-fluid EOS selector.
HA_BTU_PER_LB = JKG_TO_BTULB

def ha_calc(output, t, p, rh):
    return float(HAPropsSI(output, "T", float(t), "P", float(p), "R", float(rh)))

def ha_from_tw(output, t, p, w):
    return float(HAPropsSI(output, "T", float(t), "P", float(p), "W", float(w)))

def ha_h_display(h):
    return h_from_si(h, units)

def ha_w_display(w):
    return w * 7000 if units == "Imperial" else w * 1000

def ha_v_display(v):
    return v * 16.018463 if units == "Imperial" else v

def ha_state(t, p, rh):
    return {k: ha_calc(k, t, p, rh) for k in ("W", "H", "V", "Twb", "Tdp", "S")}

@st.cache_data(show_spinner=False, max_entries=24)
def ha_chart_curves(p, tlo, thi, n=65):
    curves = []
    for rh in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        xx, yy = [], []
        for t in np.linspace(tlo, thi, n):
            try:
                w = ha_calc("W", t, p, rh)
                if np.isfinite(w) and 0 <= w <= 0.2:
                    xx.append(t); yy.append(w)
            except (ValueError, OverflowError):
                pass
        if xx:
            curves.append((rh, xx, yy))
    return curves

if page == "Psychrometrics":
    st.title("Psychrometrics · Moist Air")
    st.caption("Uses CoolProp HAPropsSI (humid-air model), independent of the pure-fluid EOS. "
               "Pressure is absolute. Enthalpy, entropy, humidity ratio and specific volume are on a dry-air mass basis.")
    p_label = PUNIT(units)
    temp_label = TUNIT(units)
    w_label = "grains/lb dry air" if units == "Imperial" else "g/kg dry air"
    h_label = "BTU/lb dry air" if units == "Imperial" else "kJ/kg dry air"
    v_label = "ft³/lb dry air" if units == "Imperial" else "m³/kg dry air"
    t_default = 75.0 if units == "Imperial" else 24.0
    p_default = 14.696 if units == "Imperial" else 0.101325
    st.subheader("Air conditions")
    c1, c2, c3 = st.columns(3)
    with c1:
        air_p = st.number_input(f"Barometric pressure ({p_label})", min_value=0.001,
                                value=p_default, format="%.5f", key="ha_pressure")
    with c2:
        air_t = st.number_input(f"Dry-bulb temperature ({temp_label})", value=t_default,
                                key="ha_temperature")
    with c3:
        air_rh = st.slider("Relative humidity (%)", 1, 100, 50, key="ha_rh")
    p_si, t_si, rh = p_to_si(air_p, units), t_to_si(air_t, units), air_rh / 100
    state = None
    try:
        state = ha_state(t_si, p_si, rh)
        a, b, c, d = st.columns(4)
        a.metric("Wet-bulb", f"{t_from_si(state['Twb'], units):.2f} {temp_label}")
        b.metric("Dew point", f"{t_from_si(state['Tdp'], units):.2f} {temp_label}")
        c.metric("Humidity ratio", f"{ha_w_display(state['W']):.2f} {w_label}")
        d.metric("Enthalpy", f"{ha_h_display(state['H']):.2f} {h_label}")
        st.caption(f"Specific volume: {ha_v_display(state['V']):.4f} {v_label}  |  "
                   f"Entropy: {s_from_si(state['S'], units):.4f} "
                   f"{'BTU/(lb dry air·°R)' if units == 'Imperial' else 'kJ/(kg dry air·K)'}")
    except (ValueError, OverflowError) as exc:
        st.error(f"CoolProp could not calculate this moist-air state: {exc}")

    tab_chart, tab_process, tab_mix, tab_cool = st.tabs([
        "Psychrometric chart", "Heating / cooling paths", "Mixing two air streams", "Cooling & dehumidification"
    ])
    with tab_chart:
        st.write("Interactive dry-bulb versus humidity-ratio chart with constant-relative-humidity curves.")
        c1, c2 = st.columns(2)
        with c1:
            t_min = st.number_input(f"Chart minimum dry-bulb ({temp_label})", value=32.0 if units == "Imperial" else 0.0, key="ha_chart_min")
        with c2:
            t_max = st.number_input(f"Chart maximum dry-bulb ({temp_label})", value=120.0 if units == "Imperial" else 50.0, key="ha_chart_max")
        if t_max <= t_min:
            st.warning("Maximum chart temperature must exceed minimum.")
        else:
            fig = go.Figure()
            with st.spinner("Calculating humid-air chart..."):
                curves = ha_chart_curves(p_si, t_to_si(t_min, units), t_to_si(t_max, units))
            for curve_rh, temps, ws in curves:
                fig.add_trace(go.Scatter(x=[t_from_si(t, units) for t in temps],
                                         y=[ha_w_display(w) for w in ws], mode="lines",
                                         name=f"{curve_rh:.0%} RH", line={"width": 2.5 if curve_rh == 1 else 1.2},
                                         hovertemplate=f"RH: {curve_rh:.0%}<br>T: %{{x:.1f}}<br>W: %{{y:.2f}}<extra></extra>"))
            if state is not None:
                fig.add_trace(go.Scatter(x=[air_t], y=[ha_w_display(state["W"])],
                                         mode="markers", marker={"size": 12, "symbol": "diamond"},
                                         name="Selected state"))
            fig.update_layout(xaxis_title=f"Dry-bulb temperature ({temp_label})",
                              yaxis_title=f"Humidity ratio ({w_label})", height=620,
                              hovermode="closest", legend_title="Relative humidity")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Curves are calculated from HAPropsSI at the selected barometric pressure. "
                       "This is an RH chart, not a full ASHRAE-style chart with enthalpy, wet-bulb and volume isolines.")

    with tab_process:
        st.write("Constant-humidity-ratio sensible heating or cooling from the selected inlet state. "
                 "If cooling reaches saturation, the path stops at the dew point; use the dehumidification tab for further cooling.")
        target_t = st.number_input(f"Target dry-bulb ({temp_label})", value=95.0 if units == "Imperial" else 35.0,
                                   key="ha_target_t")
        if state is not None:
            try:
                inlet_w = state["W"]
                target_si = t_to_si(target_t, units)
                end_si = max(target_si, state["Tdp"]) if target_si < t_si else target_si
                end_h = ha_from_tw("H", end_si, p_si, inlet_w)
                duty = end_h - state["H"]
                st.metric("Sensible enthalpy change per dry-air mass", f"{ha_h_display(duty):+.2f} {h_label}")
                if end_si != target_si:
                    st.warning("Target is below inlet dew point; cooling at constant humidity ratio ends at saturation.")
                xs = np.linspace(t_si, end_si, 25)
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=[t_from_si(x, units) for x in xs],
                                          y=[ha_w_display(inlet_w)] * len(xs), mode="lines+markers",
                                          name="Sensible process"))
                fig2.update_layout(xaxis_title=f"Dry-bulb ({temp_label})", yaxis_title=f"Humidity ratio ({w_label})")
                st.plotly_chart(fig2, use_container_width=True)
            except (ValueError, OverflowError) as exc:
                st.error(f"Process calculation unavailable: {exc}")

    with tab_mix:
        st.write("Adiabatic mixing at the same pressure, with dry-air mass flow rates. "
                 "Uses conservation of dry air, water vapor and moist-air enthalpy; assumes no condensation during mixing.")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Stream 1**")
            t1 = st.number_input(f"Temperature 1 ({temp_label})", value=t_default, key="ha_t1")
            r1 = st.slider("Relative humidity 1 (%)", 1, 100, 50, key="ha_r1")
            m1 = st.number_input("Dry-air mass flow 1 (relative or mass/time)", min_value=0.001, value=1.0, key="ha_m1")
        with c2:
            st.markdown("**Stream 2**")
            t2 = st.number_input(f"Temperature 2 ({temp_label})", value=95.0 if units == "Imperial" else 35.0, key="ha_t2")
            r2 = st.slider("Relative humidity 2 (%)", 1, 100, 30, key="ha_r2")
            m2 = st.number_input("Dry-air mass flow 2 (same units)", min_value=0.001, value=1.0, key="ha_m2")
        try:
            s1 = ha_state(t_to_si(t1, units), p_si, r1 / 100)
            s2 = ha_state(t_to_si(t2, units), p_si, r2 / 100)
            wm = (m1*s1["W"] + m2*s2["W"]) / (m1+m2)
            hm = (m1*s1["H"] + m2*s2["H"]) / (m1+m2)
            tm = float(HAPropsSI("T", "P", p_si, "W", wm, "H", hm))
            rm = float(HAPropsSI("R", "P", p_si, "W", wm, "T", tm))
            if rm > 1.0001:
                st.warning("Calculated mixture is supersaturated. Condensation and an energy balance including liquid water are needed; this simple model is not valid.")
            else:
                a, b, c = st.columns(3)
                a.metric("Mixed dry-bulb", f"{t_from_si(tm, units):.2f} {temp_label}")
                b.metric("Mixed relative humidity", f"{100*rm:.1f}%")
                c.metric("Mixed humidity ratio", f"{ha_w_display(wm):.2f} {w_label}")
                st.metric("Mixed enthalpy", f"{ha_h_display(hm):.2f} {h_label}")
        except (ValueError, OverflowError) as exc:
            st.error(f"Mixing calculation unavailable: {exc}")

    with tab_cool:
        st.write("Idealized cooling-coil outlet at the specified dry-bulb temperature. "
                 "Below inlet dew point, assumes saturated outlet (100% RH). "
                 "No bypass factor or coil efficiency is included.")
        outlet_t = st.number_input(f"Coil outlet dry-bulb ({temp_label})", value=55.0 if units == "Imperial" else 13.0,
                                    key="ha_coil_outlet")
        if state is not None:
            try:
                to_si = t_to_si(outlet_t, units)
                if to_si >= t_si:
                    st.info("Outlet is not cooler than inlet; this is not a cooling process.")
                else:
                    if to_si < state["Tdp"]:
                        wo = ha_calc("W", to_si, p_si, 1.0)
                        ho = ha_calc("H", to_si, p_si, 1.0)
                        out_rh = 100.0
                    else:
                        wo = state["W"]
                        ho = ha_from_tw("H", to_si, p_si, wo)
                        out_rh = 100 * ha_from_tw("R", to_si, p_si, wo)
                    removed = max(0.0, state["W"] - wo)
                    a, b, c = st.columns(3)
                    a.metric("Cooling enthalpy change", f"{ha_h_display(ho - state['H']):+.2f} {h_label}")
                    b.metric("Condensate per dry-air mass", f"{ha_w_display(removed):.2f} {w_label}")
                    c.metric("Outlet relative humidity", f"{out_rh:.1f}%")
                    st.caption("For total cooling load with condensate leaving the coil, account for the "
                               "enthalpy of drained liquid water. The enthalpy difference shown is a useful "
                               "moist-air approximation, not a complete coil energy balance.")
            except (ValueError, OverflowError) as exc:
                st.error(f"Cooling calculation unavailable: {exc}")


# -------------------- PIPE FROST GROWTH (SCREENING MODEL) --------------------
def frost_ice_pressure(t_kelvin):
    """Murphy & Koop (2005) saturation vapor pressure over ice, Pa (110–273.16 K)."""
    return float(np.exp(9.550426 - 5723.265 / t_kelvin + 3.53068 * np.log(t_kelvin) - 0.00728332 * t_kelvin))


def frost_air_vapor_pressure(t_kelvin, p_pa, rh):
    """Use CoolProp humid-air humidity ratio to obtain approximate water partial pressure."""
    w = float(HAPropsSI("W", "T", t_kelvin, "P", p_pa, "R", rh))
    return p_pa * w / (0.621945 + w)


def frost_simulate(diameter_m, pipe_t, air_t, p_pa, rh, air_speed, hours,
                   frost_density=180.0, frost_k=0.15):
    """Simplified cylinder heat/mass-transfer frost model; constant frost properties.

    Airflow uses Churchill–Bernstein cylinder correlation with a heat/mass analogy.
    Assumes constant bare pipe wall temperature, uniform frost and no liquid runoff.
    Radiation, frost densification, condensate freezing and axial effects omitted.
    """
    if not (173.15 < pipe_t < 273.15 and 173.15 < air_t < 373.15):
        raise ValueError("Choose a subfreezing pipe temperature and air temperature within -100 to 100°C.")
    r0 = diameter_m / 2.0
    rv = 461.5
    pv_air = frost_air_vapor_pressure(air_t, p_pa, rh)
    # Approximate air properties at film temperature (constant in this screening model)
    tfilm = (pipe_t + air_t) / 2
    rho_air = p_pa / (287.05 * tfilm)
    mu = 1.8e-5 * (tfilm / 300.0)**0.7
    k_air = 0.026 * (tfilm / 300.0)**0.85
    cp_air = 1006.0
    d_v = 2.5e-5 * (tfilm / 298.15)**1.8 * (101325.0 / p_pa)
    pr = cp_air * mu / k_air
    sc = mu / (rho_air * d_v)
    def cylinder_correlation(re, dimensionless):
        return (0.3 + (0.62 * np.sqrt(max(re, 0)) * dimensionless**(1/3) /
                (1 + (0.4 / dimensionless)**(2/3))**0.25) *
                (1 + (re / 282000)**(5/8))**(4/5))
    def rates(radius, surface_t):
        diameter = 2 * radius
        # 0.1 m/s floor avoids the forced-convection correlation becoming zero;
        # this is NOT a full natural-convection model.
        re = rho_air * max(air_speed, 0.1) * diameter / mu
        nu = cylinder_correlation(re, pr)
        sh = cylinder_correlation(re, sc)
        h = nu * k_air / diameter
        hm = sh * d_v / diameter
        pv_surface = frost_ice_pressure(surface_t)
        mass_flux = max(0.0, hm * (pv_air / (rv * air_t) - pv_surface / (rv * surface_t)))
        return h, mass_flux
    total_seconds = hours * 3600.0
    dt = max(10.0, total_seconds / 360.0)
    elapsed = 0.0
    frost_mass_per_m = 0.0
    rows = []
    while elapsed <= total_seconds + 0.001:
        radius = np.sqrt(r0**2 + frost_mass_per_m / (np.pi * frost_density))
        resistance = np.log(radius / r0) / (2 * np.pi * frost_k) if radius > r0 else 0.0
        def residual(ts):
            h, mdot = rates(radius, ts)
            heat_in = 2 * np.pi * radius * (h * (air_t - ts) + mdot * 2.834e6)
            heat_out = (ts - pipe_t) / resistance if resistance > 1e-12 else 1e20 * (ts - pipe_t)
            return heat_in - heat_out
        low, high = pipe_t, max(pipe_t, air_t)
        for _ in range(45):
            mid = (low + high) / 2
            if residual(mid) > 0:
                low = mid
            else:
                high = mid
        surface_t = (low + high) / 2
        h, mass_flux = rates(radius, surface_t)
        deposition_per_m_s = mass_flux * 2 * np.pi * radius
        rows.append({"Hours": elapsed / 3600, "Thickness_mm": (radius - r0) * 1000,
                     "Frost_surface_K": surface_t, "Frost_kg_per_m": frost_mass_per_m,
                     "Deposition_g_per_m_h": deposition_per_m_s * 3.6e6})
        if elapsed >= total_seconds: break
        step = min(dt, total_seconds - elapsed)
        frost_mass_per_m += deposition_per_m_s * step
        elapsed += step
    return pd.DataFrame(rows)


if page == "Pipe Frost Growth":
    st.title("Pipe Frost Growth · Animated Cross-Section")
    st.caption("Preliminary vapor-deposition frost estimate for an exposed cold pipe. "
               "Not a validated icing prediction or an engineering design basis.")
    st.info("Assumes a constant pipe outside-wall temperature, uniform porous frost, and "
            "forced air crossflow. Liquid water freezing, rain, dripping, natural convection, "
            "frost compaction and radiation are not modeled.")
    c1, c2, c3 = st.columns(3)
    imperial = units == "Imperial"
    with c1:
        od = st.number_input("Bare pipe outside diameter (in)" if imperial else "Bare pipe outside diameter (mm)",
                             min_value=0.1, value=2.375 if imperial else 60.3, step=0.1)
        pipe_temp = st.number_input(f"Pipe outside surface temperature ({TUNIT(units)})",
                                    value=-20.0 if imperial else -28.9, step=1.0)
        exposure = st.slider("Exposure time (hours)", 1, 168, 24)
    with c2:
        ambient = st.number_input(f"Ambient dry-bulb temperature ({TUNIT(units)})",
                                  value=80.0 if imperial else 26.7, step=1.0)
        humidity = st.slider("Relative humidity (%)", 1, 100, 70)
        pressure = st.number_input("Atmospheric pressure (psia)" if imperial else "Atmospheric pressure (kPa abs)",
                                   min_value=1.0, value=14.696 if imperial else 101.325)
    with c3:
        speed = st.number_input("Air speed across pipe (mph)" if imperial else "Air speed across pipe (m/s)",
                                min_value=0.1, value=2.0 if imperial else 0.9, step=0.1)
        density = st.number_input("Assumed frost density (kg/m³)", min_value=50.0, max_value=800.0,
                                  value=180.0, step=10.0)
        conductivity = st.number_input("Assumed frost conductivity (W/m·K)", min_value=0.03,
                                       max_value=2.0, value=0.15, step=0.01, format="%.2f")
    try:
        od_m = od * 0.0254 if imperial else od / 1000
        p_pa = pressure * PSI_TO_PA if imperial else pressure * 1000
        velocity = speed * 0.44704 if imperial else speed
        pipe_k, air_k = t_to_si(pipe_temp, units), t_to_si(ambient, units)
        if air_k <= pipe_k:
            st.warning("Ambient temperature must exceed pipe surface temperature for this simplified heat-transfer model.")
        else:
            results = frost_simulate(od_m, pipe_k, air_k, p_pa, humidity / 100,
                                     velocity, exposure, density, conductivity)
            last = results.iloc[-1]
            a, b, c = st.columns(3)
            thickness = last["Thickness_mm"] / 25.4 if imperial else last["Thickness_mm"]
            a.metric("Predicted frost thickness", f"{thickness:.3f} {'in' if imperial else 'mm'}")
            b.metric("Final frost surface temperature", f"{t_from_si(last['Frost_surface_K'], units):.1f} {TUNIT(units)}")
            c.metric("Frost mass per pipe length", f"{last['Frost_kg_per_m']:.3f} kg/m")
            st.caption("Approximate result; frost density, conductivity, convection and phase-change mechanisms "
                       "can substantially change actual thickness.")
            st.subheader("Animated pipe cross-section")
            st.caption("Press ▶ to animate frost growth. The time slider lets you inspect an individual frame.")
            frame_hour = st.slider("Cross-section at elapsed time (hours)", 0.0, float(exposure),
                                   float(exposure), step=0.25)
            r0_mm = od_m * 500
            max_outer = r0_mm + max(0.5, float(results['Thickness_mm'].max()))
            axis_lim = max_outer * 1.18
            theta = np.linspace(0, 2*np.pi, 150)
            def section_figure(thick_mm):
                outer = r0_mm + thick_mm
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=outer*np.cos(theta), y=outer*np.sin(theta),
                    fill="toself", fillcolor="rgba(153,214,243,0.72)",
                    line=dict(color="#5DA9D1", width=2), name="Frost outer edge", hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=r0_mm*np.cos(theta), y=r0_mm*np.sin(theta),
                    fill="toself", fillcolor="#71818B", line=dict(color="#3E4D56", width=2),
                    name="Pipe OD", hoverinfo="skip"))
                fig.update_layout(xaxis=dict(range=[-axis_lim,axis_lim], visible=False, scaleanchor="y"),
                                  yaxis=dict(range=[-axis_lim,axis_lim], visible=False),
                                  margin=dict(l=15,r=15,t=45,b=10), height=430,
                                  title=f"Frost layer: {thick_mm:.2f} mm", showlegend=True)
                return fig
            selected_thickness = float(np.interp(frame_hour, results["Hours"], results["Thickness_mm"]))
            left, right = st.columns([1,1])
            with left:
                fig = section_figure(selected_thickness)
                # Plotly animation frames use the same two traces and shared axis bounds.
                samples = np.linspace(0, len(results)-1, min(45, len(results))).astype(int)
                frames = []
                for idx in samples:
                    row = results.iloc[idx]
                    rr = r0_mm + float(row["Thickness_mm"])
                    frames.append(go.Frame(name=f"{row['Hours']:.1f} h", data=[
                        go.Scatter(x=rr*np.cos(theta), y=rr*np.sin(theta),
                                   fill="toself", fillcolor="rgba(153,214,243,0.72)",
                                   line=dict(color="#5DA9D1", width=2)),
                        go.Scatter(x=r0_mm*np.cos(theta), y=r0_mm*np.sin(theta),
                                   fill="toself", fillcolor="#71818B", line=dict(color="#3E4D56", width=2))],
                        layout=go.Layout(title_text=f"Elapsed: {row['Hours']:.1f} h · Frost: {row['Thickness_mm']:.2f} mm")))
                fig.frames = frames
                fig.update_layout(updatemenus=[dict(type="buttons", showactive=False, x=0.05, y=0,
                    buttons=[dict(label="▶ Play", method="animate", args=[None,
                        {"frame":{"duration":100,"redraw":True},"fromcurrent":True,
                         "transition":{"duration":0}}]),
                             dict(label="❚❚ Pause", method="animate", args=[[None],
                                 {"frame":{"duration":0,"redraw":False},"mode":"immediate"}])])])
                st.plotly_chart(fig, use_container_width=True)
            with right:
                fig2 = go.Figure()
                values = results["Thickness_mm"] / 25.4 if imperial else results["Thickness_mm"]
                fig2.add_trace(go.Scatter(x=results["Hours"], y=values, mode="lines",
                                          name="Frost thickness"))
                fig2.add_vline(x=frame_hour, line_dash="dash", line_color="gray")
                fig2.update_layout(title="Frost thickness vs. time", xaxis_title="Hours",
                    yaxis_title="Frost thickness (in)" if imperial else "Frost thickness (mm)", height=430)
                st.plotly_chart(fig2, use_container_width=True)
            with st.expander("Calculation data and model assumptions"):
                st.dataframe(results, use_container_width=True)
                st.download_button("Download CSV", data=results.to_csv(index=False),
                                   file_name="thermotools_pipe_frost.csv", mime="text/csv")
                st.markdown("**Model:** moisture transport by a cylinder crossflow Sherwood-number "
                            "correlation; equilibrium vapor pressure over ice at the frost surface; "
                            "cylindrical conduction through frost; convective sensible heat and "
                            "deposition latent heat. Frost density and thermal conductivity are "
                            "fixed. Air properties are approximate. Wind speed below 0.1 m/s is "
                            "floored at 0.1 m/s; natural convection is not explicitly solved.")
    except (ValueError, OverflowError, ZeroDivisionError) as exc:
        st.error(f"Frost calculation unavailable: {exc}")


# -------------------- MECHANICAL ENGINEERING: FATIGUE --------------------
if page == "Cyclic & Fatigue Loading":
    st.title("Mechanical Engineering | Cyclic & Fatigue Loading")
    st.info("Educational stress-life (S–N) screening for a uniaxial, predominantly elastic, constant-amplitude stress cycle. Not an ASME fatigue assessment, pressure-boundary qualification, or a prediction of safe remaining life.")
    st.caption("All strength and S–N inputs must be supplied from applicable, traceable material/test data. Grade names alone do not establish fatigue performance, especially for welded, corroded, notched, or elevated-temperature equipment.")

    f_unit = "ksi" if units == "Imperial" else "MPa"
    to_mpa = 6.894757293 if units == "Imperial" else 1.0
    from_mpa = 1.0 / to_mpa
    st.subheader("Material and loading")
    a, b = st.columns(2)
    with a:
        material_name = st.text_input("Material / component description", "Carbon steel — illustrative inputs")
        sut = st.number_input(f"Ultimate tensile strength, Sᵤ ({f_unit})", min_value=0.001, value=70.0 if units == "Imperial" else 480.0, step=1.0)
        sy = st.number_input(f"Yield strength, Sᵧ ({f_unit})", min_value=0.001, value=40.0 if units == "Imperial" else 275.0, step=1.0)
        correction = st.selectbox("Mean-stress correction", ["Modified Goodman", "Soderberg", "Gerber", "None"])
    with b:
        smax = st.number_input(f"Maximum local stress, σmax ({f_unit})", value=20.0 if units == "Imperial" else 140.0, step=1.0)
        smin = st.number_input(f"Minimum local stress, σmin ({f_unit})", value=0.0, step=1.0)
        frequency = st.number_input("Cycles per hour (for time conversion)", min_value=0.0001, value=60.0, step=1.0)
        cycles_target = st.number_input("Target service cycles", min_value=1.0, value=1e6, format="%.0f")

    st.subheader("S–N data (fully reversed, R = −1)")
    st.caption("Enter two experimentally supported stress-amplitude points for the relevant component/detail, in the same units as above. The app interpolates/extrapolates a straight line in log(stress) versus log(cycles). It does NOT assume an endurance limit.")
    c1, c2, c3, c4 = st.columns(4)
    with c1: n1 = st.number_input("Cycles N₁", min_value=1.0, value=1e3, format="%.0f")
    with c2: sn1 = st.number_input(f"Amplitude S₁ ({f_unit})", min_value=0.001, value=55.0 if units == "Imperial" else 380.0)
    with c3: n2 = st.number_input("Cycles N₂", min_value=1.0, value=1e6, format="%.0f")
    with c4: sn2 = st.number_input(f"Amplitude S₂ ({f_unit})", min_value=0.001, value=24.0 if units == "Imperial" else 165.0)

    if smax <= smin:
        st.error("Maximum stress must exceed minimum stress.")
    elif n2 <= n1 or sn1 <= sn2:
        st.error("S–N points must have N₂ > N₁ and S₁ > S₂ for a decreasing stress-life curve.")
    elif sy > sut:
        st.error("Yield strength should not exceed ultimate tensile strength for this screening model.")
    else:
        mean = (smax + smin) / 2.0
        amplitude = (smax - smin) / 2.0
        stress_range = smax - smin
        ratio = smin / smax if abs(smax) > 1e-12 else float('nan')
        # Use tensile mean-stress correction only; do not credit compressive mean stress.
        positive_mean = max(mean, 0.0)
        strength = sut if correction != "Soderberg" else sy
        fraction = positive_mean / strength
        if correction != "None" and fraction >= 1:
            st.error("Tensile mean stress meets or exceeds the strength used in this correction. The fatigue-life equation is not valid here.")
        else:
            if correction == "Modified Goodman" or correction == "Soderberg":
                factor = 1.0 - fraction
            elif correction == "Gerber":
                factor = 1.0 - fraction**2
            else:
                factor = 1.0
            equivalent = amplitude / factor
            exponent = np.log(sn2 / sn1) / np.log(n2 / n1)
            coeff = sn1 / (n1 ** exponent)
            predicted_cycles = float((equivalent / coeff) ** (1.0 / exponent))
            stress_at_target = float(coeff * cycles_target ** exponent)
            allowed_amplitude = stress_at_target * factor
            utilization = amplitude / allowed_amplitude
            st.subheader("Calculated results")
            x1, x2, x3, x4 = st.columns(4)
            x1.metric("Alternating stress", f"{amplitude:,.2f} {f_unit}")
            x2.metric("Mean stress", f"{mean:,.2f} {f_unit}")
            x3.metric("Corrected amplitude", f"{equivalent:,.2f} {f_unit}")
            x4.metric("Estimated cycles", f"{predicted_cycles:,.3g}")
            y1, y2, y3 = st.columns(3)
            y1.metric("Target-cycle utilization", f"{utilization:.3f}")
            y2.metric("Target allowable amplitude", f"{allowed_amplitude:,.2f} {f_unit}")
            y3.metric("Nominal time to predicted cycles", f"{predicted_cycles/frequency:,.1f} h")
            if smax > sy or smin < -sy:
                st.warning("The specified stress cycle reaches/exceeds the nominal yield strength in tension or compression. Elastic stress-life assumptions may not apply; investigate strain-life and local plasticity.")
            if predicted_cycles < n1 or predicted_cycles > n2:
                st.warning("Predicted life is OUTSIDE the supplied S–N data range (extrapolation). Do not treat this result as a validated life prediction.")
            if mean < 0:
                st.caption("Conservative convention: compressive mean stress receives no beneficial correction.")
            tab1, tab2, tab3 = st.tabs(["Stress cycle", "S–N curve", "Mean-stress diagram"])
            with tab1:
                phase = np.linspace(0, 4*np.pi, 300)
                sigma = mean + amplitude*np.cos(phase)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=phase/(2*np.pi), y=sigma, name="Applied stress", mode="lines"))
                fig.add_hline(y=mean, line_dash="dash", annotation_text="Mean stress")
                fig.update_layout(title="Two representative stress cycles", xaxis_title="Cycles", yaxis_title=f"Stress ({f_unit})", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
                st.write(f"Stress range: **{stress_range:,.3f} {f_unit}** · Stress ratio R: **{ratio:.3f}**" if np.isfinite(ratio) else f"Stress range: **{stress_range:,.3f} {f_unit}** · R undefined (σmax = 0)")
            with tab2:
                cycle_grid = np.geomspace(max(1.0, n1/5), n2*5, 200)
                strength_grid = coeff * cycle_grid**exponent
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=cycle_grid, y=strength_grid, name="Input-derived S–N fit", mode="lines"))
                fig.add_trace(go.Scatter(x=[n1,n2], y=[sn1,sn2], mode="markers", name="Supplied S–N points", marker=dict(size=11)))
                fig.add_trace(go.Scatter(x=[predicted_cycles], y=[equivalent], mode="markers", name="Corrected operating point", marker=dict(size=13, symbol="diamond")))
                fig.update_layout(title="Stress amplitude vs cycles (R = −1 reference)", xaxis_title="Cycles to failure", yaxis_title=f"Stress amplitude ({f_unit})", xaxis_type="log", yaxis_type="log", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Extrapolation beyond the two supplied points is shown for illustration, not validated fatigue performance.")
            with tab3:
                means = np.linspace(0, min(sut, max(sut, sy)), 200)
                fig = go.Figure()
                for label, strength_limit, method in [("Goodman", sut, "linear"), ("Soderberg", sy, "linear"), ("Gerber", sut, "quadratic")]:
                    xx = np.linspace(0, strength_limit, 200)
                    yy = stress_at_target * (1-(xx/strength_limit)**(2 if method=="quadratic" else 1))
                    fig.add_trace(go.Scatter(x=xx, y=yy, mode="lines", name=label))
                fig.add_trace(go.Scatter(x=[mean], y=[amplitude], mode="markers", name="Operating point", marker=dict(size=12, symbol="diamond")))
                fig.update_layout(title=f"Mean stress vs allowable alternating stress at {cycles_target:,.0f} cycles", xaxis_title=f"Mean stress ({f_unit})", yaxis_title=f"Alternating stress ({f_unit})", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
                st.caption("The curves are illustrative mean-stress correction envelopes derived from the supplied S–N fit, not code-based acceptance boundaries.")
            with st.expander("Method, assumptions, and equations"):
                st.latex(r"\sigma_a = (\sigma_{max}-\sigma_{min})/2,\qquad \sigma_m = (\sigma_{max}+\sigma_{min})/2")
                st.latex(r"S_a(N)=A N^b")
                st.latex(r"\sigma_{a,eq}=\frac{\sigma_a}{1-\sigma_m/S_u}\quad\text{(Goodman, tensile mean stress)}")
                st.write("Soderberg substitutes yield strength for ultimate strength; Gerber uses a squared mean-stress ratio. The input stress must already represent the relevant local fatigue-driving stress; geometry and notch effects are not inferred. Elastic constant-amplitude uniaxial loading is assumed.")
                st.write("This tool excludes weld-detail fatigue classes, multiaxial loading, corrosion fatigue, creep-fatigue, temperature dependence, surface finish, residual stress, reliability factors, and rainflow counting. Validate against the relevant material/component test data and engineering code.")
            output = pd.DataFrame([{"Material":material_name,"Correction":correction,"Stress unit":f_unit,"Maximum stress":smax,"Minimum stress":smin,"Alternating stress":amplitude,"Mean stress":mean,"Corrected amplitude":equivalent,"Estimated cycles":predicted_cycles,"Target cycles":cycles_target,"Utilization":utilization,"SN N1":n1,"SN S1":sn1,"SN N2":n2,"SN S2":sn2}])
            st.download_button("Download fatigue screening results (CSV)", output.to_csv(index=False), file_name="thermotools_fatigue.csv", mime="text/csv")

# -------------------- HOME --------------------
if page == "Home":
    st.title("ThermoTools")
    st.subheader(f"{FLUID} Thermodynamics Toolkit")
    st.write(f"An engineering calculation environment for exploring the thermodynamic behavior of {FLUID}.")
    st.markdown("- **Pressure-Enthalpy Diagram:** Saturation envelope, state points, and process paths.\n- **State Point Calculator:** Properties from pressure and temperature.\n- **Isenthalpic Expansion:** Final temperature at constant enthalpy.\n- **Joule-Thomson Calculator:** Local Joule-Thomson coefficient.")
    st.info(f"Pure-fluid calculations use CoolProp {EOS}. Mixture tools use HEOS. Validate results before plant design or operating decisions.")

# -------------------- P-H DIAGRAM --------------------
elif page == "Pressure-Enthalpy Diagram":
    st.title(f"{FLUID} Pressure-Enthalpy Diagram")
    st.write(f"Explore the thermodynamic behavior of {FLUID}. Add state points and calculated process paths.")
    try:
        pc, tc = PropsSI("Pcrit", FLUID), PropsSI("Tcrit", FLUID)
        try:
            hc = PropsSI("H", "P", pc * 1.0001, "T", tc * 1.0001, FLUID)
        except Exception:
            hc = None
        ptriple = PropsSI("ptriple", PURE_FLUID)
    except Exception as e:
        st.error(f"Could not retrieve model limits for {FLUID}: {e}"); st.stop()
    a,b,c = st.columns(3)
    with a: pmin_d = st.number_input(f"Minimum pressure ({PUNIT(units)})", min_value=0.1 if units=="Imperial" else .001, value=1.0 if units=="Imperial" else .01)
    with b: pmax_d = st.number_input(f"Maximum pressure ({PUNIT(units)})", min_value=10.0 if units=="Imperial" else .1, value=14000.0 if units=="Imperial" else 90.0)
    with c: resolution = st.select_slider("Curve resolution", ["Low","Medium","High"], value="Medium")
    n = {"Low":35,"Medium":75,"High":140}[resolution]
    pmin, pmax = p_to_si(pmin_d,units), p_to_si(pmax_d,units)
    pmin_used, pmax_used = max(pmin, 1.0), pmax
    if pmin >= pmax or pmin_used >= pmax_used:
        st.error("The selected pressure range is invalid or outside the CoolProp model range."); st.stop()
    st.caption("Property calculations outside the fluid model’s valid range are skipped when drawing curves.")
    pressures = np.geomspace(pmin_used,pmax_used,n)
    show_q = st.checkbox("Show vapor quality lines",True)
    show_t = st.checkbox("Show temperature lines",True)
    show_crit = st.checkbox("Show critical point",True)
    fig = go.Figure()
    # Pure-fluid saturation curves; mixtures will need a different phase-envelope method.
    psat0, psat1 = max(ptriple*1.01,1.0), pc*.999
    if psat0 < psat1:
        psats = np.geomspace(psat0,psat1,n)
        hl,hv,py = [],[],[]
        for p in psats:
            try:
                hl.append(h_from_si(PropsSI("H","P",p,"Q",0,FLUID),units))
                hv.append(h_from_si(PropsSI("H","P",p,"Q",1,FLUID),units))
                py.append(p_from_si(p,units))
            except Exception: continue
        if py:
            fig.add_trace(go.Scatter(x=hl,y=py,mode="lines",name="Saturated liquid",line=dict(width=3)))
            fig.add_trace(go.Scatter(x=hv,y=py,mode="lines",name="Saturated vapor",line=dict(width=3)))
        if show_q:
            for q in np.arange(.1,1.0,.1):
                x,y=[],[]
                for p in psats:
                    try: x.append(h_from_si(PropsSI("H","P",p,"Q",float(q),FLUID),units))
                    except Exception: x.append(None)
                    y.append(p_from_si(p,units))
                fig.add_trace(go.Scatter(x=x,y=y,mode="lines",name=f"Quality {q:.0%}",line=dict(width=1,dash="dot"),connectgaps=False))
    else:
        st.info("No saturation dome is available for this fluid in the model range.")
    if show_t:
        temps = [-300,-200,-100,0,50,100,150,200,250,300,400,500,600,700,800] if units=="Imperial" else [-200,-150,-100,-50,0,25,50,75,100,150,200,250,300,400,500]
        for td in temps:
            tk = f_to_k(td) if units=="Imperial" else td+273.15
            if tk <= 0: continue
            x,y=[],[]
            for p in pressures:
                try: x.append(h_from_si(PropsSI("H","P",p,"T",tk,FLUID),units))
                except Exception: x.append(None)
                y.append(p_from_si(p,units))
            fig.add_trace(go.Scatter(x=x,y=y,mode="lines",name=f"{td:g}{TUNIT(units)}",line=dict(width=1,dash="dash"),connectgaps=False))
    if show_crit and hc is not None:
        fig.add_trace(go.Scatter(x=[h_from_si(hc,units)],y=[p_from_si(pc,units)],mode="markers+text",name="Critical point",text=["Critical point"],textposition="top right",marker=dict(size=12,symbol="star")))

    st.divider(); st.subheader("State points and process paths")
    if "ph_points" not in st.session_state: st.session_state.ph_points=[]
    if "ph_paths" not in st.session_state: st.session_state.ph_paths=[]
    with st.expander("Add a state point",expanded=True):
        with st.form("add_ph_point"):
            name = st.text_input("Point name",value=f"Point {len(st.session_state.ph_points)+1}")
            x1,x2=st.columns(2)
            with x1: pdsp=st.number_input(PLABEL(units),min_value=.001,value=500.0 if units=="Imperial" else 3.447)
            with x2: tdsp=st.number_input(TLABEL(units),value=100.0 if units=="Imperial" else 37.78)
            add=st.form_submit_button("Calculate and add point")
            if add:
                try:
                    state=state_pt(p_to_si(pdsp,units),f_to_k(tdsp) if units=="Imperial" else tdsp+273.15)
                    state["name"]=name.strip() or f"Point {len(st.session_state.ph_points)+1}"
                    if any(z["name"]==state["name"] for z in st.session_state.ph_points): raise ValueError("That point name already exists.")
                    st.session_state.ph_points.append(state); st.success(f"{state['name']} added.")
                except Exception as e: st.error(f"Could not calculate state: {e}")

    if st.session_state.ph_points:
        with st.expander("Create a calculated process path",expanded=True):
            opts=list(range(len(st.session_state.ph_points)))
            start_i=st.selectbox("Starting state point",opts,format_func=lambda i:f"{i+1}: {st.session_state.ph_points[i]['name']}")
            start=st.session_state.ph_points[start_i]
            choices=["Isenthalpic expansion","Isentropic compression","Isentropic expansion","Isobaric heating","Isobaric cooling","Isothermal compression","Isothermal expansion","Isochoric heating","Isochoric cooling"]
            proc=st.selectbox("Process type",choices)
            pressure_procs=choices[:3]+choices[5:]
            if proc in pressure_procs:
                startpd=p_from_si(start["P"],units)
                default=startpd*.5 if proc in ["Isenthalpic expansion","Isentropic expansion","Isothermal expansion"] else startpd*2
                endp=st.number_input(f"Ending pressure ({PUNIT(units)})",min_value=.001,value=float(max(.001,default)),key="path_end_p")
            else:
                starttd=f_to_k(start["T"])*9/5-459.67 if units=="Imperial" else start["T"]-273.15
                off=50 if units=="Imperial" else 25
                endt=st.number_input(f"Ending temperature ({TUNIT(units)})",value=float(starttd+off if proc=="Isobaric heating" else starttd-off),key="path_end_t")
            endname=st.text_input("Name for ending point",value=f"Point {len(st.session_state.ph_points)+1}")
            if st.button("Calculate and plot process"):
                try:
                    p1,t1,h1,s1,d1=start["P"],start["T"],start["H"],start["S"],start["D"]
                    if proc in pressure_procs:
                        p2=p_to_si(endp,units)
                        if proc in ["Isenthalpic expansion","Isentropic expansion","Isothermal expansion"] and p2>=p1: raise ValueError("This process requires a lower ending pressure.")
                        if proc in ["Isentropic compression","Isothermal compression"] and p2<=p1: raise ValueError("This process requires a higher ending pressure.")
                        pp=np.geomspace(p1,p2,50)
                        if proc=="Isenthalpic expansion":
                            t2=PropsSI("T","P",p2,"H",h1,FLUID); hh=[h1]*len(pp)
                        elif proc in ["Isentropic compression","Isentropic expansion"]:
                            t2=PropsSI("T","P",p2,"S",s1,FLUID); hh=[PropsSI("H","P",p,"S",s1,FLUID) for p in pp]
                        elif proc in ["Isothermal compression","Isothermal expansion"]:
                            t2=t1; hh=[PropsSI("H","P",p,"T",t1,FLUID) for p in pp]
                        else:
                            t2=PropsSI("T","P",p2,"D",d1,FLUID); hh=[]
                            for p in pp:
                                tt=PropsSI("T","P",p,"D",d1,FLUID); hh.append(PropsSI("H","P",p,"T",tt,FLUID))
                    else:
                        p2=p1; t2=f_to_k(endt) if units=="Imperial" else endt+273.15
                        if t2<=0: raise ValueError("Temperature must be above absolute zero.")
                        if proc=="Isobaric heating" and t2<=t1: raise ValueError("Heating requires a higher ending temperature.")
                        if proc=="Isobaric cooling" and t2>=t1: raise ValueError("Cooling requires a lower ending temperature.")
                        tt=np.linspace(t1,t2,50); pp=[p1]*len(tt); hh=[PropsSI("H","P",p1,"T",t,FLUID) for t in tt]
                    endstate=state_pt(p2,t2); endstate["name"]=endname.strip() or f"Point {len(st.session_state.ph_points)+1}"
                    if any(z["name"]==endstate["name"] for z in st.session_state.ph_points): raise ValueError("That point name already exists.")
                    st.session_state.ph_points.append(endstate)
                    st.session_state.ph_paths.append({"start":start["name"],"end":endstate["name"],"type":proc,"p":list(pp),"h":list(hh)})
                    st.success(f"{proc} calculated; added {endstate['name']}."); st.rerun()
                except Exception as e: st.error(f"Could not calculate process: {e}")

    lookup={z["name"]:z for z in st.session_state.ph_points}
    for path in st.session_state.ph_paths:
        if path["start"] not in lookup or path["end"] not in lookup: continue
        xx=[h_from_si(v,units) for v in path["h"]]; yy=[p_from_si(v,units) for v in path["p"]]
        fig.add_trace(go.Scatter(x=xx,y=yy,mode="lines",name=f"{path['start']} → {path['end']} ({path['type']})",line=dict(width=3)))
        if len(xx)>1: fig.add_annotation(x=xx[-1],y=yy[-1],ax=xx[-2],ay=yy[-2],xref="x",yref="y",axref="x",ayref="y",showarrow=True,arrowhead=3)
    if st.session_state.ph_points:
        xx=[h_from_si(z["H"],units) for z in st.session_state.ph_points]; yy=[p_from_si(z["P"],units) for z in st.session_state.ph_points]
        hover=[f"{z['name']}<br>T: {t_from_si(z['T'],units):.2f} {TUNIT(units)}<br>P: {p_from_si(z['P'],units):.2f} {PUNIT(units)}<br>H: {h_from_si(z['H'],units):.3f} {HUNIT(units)}<br>Phase: {z['phase']}" for z in st.session_state.ph_points]
        fig.add_trace(go.Scatter(x=xx,y=yy,mode="markers+text",name="User state points",text=[z["name"] for z in st.session_state.ph_points],textposition="top right",marker=dict(size=11,symbol="diamond"),hovertext=hover,hoverinfo="text"))
    fig.update_layout(title=f"{FLUID} Pressure-Enthalpy Diagram",xaxis_title=HLABEL(units),yaxis_title=PLABEL(units),yaxis_type="log",template="plotly_dark",height=850,hovermode="closest",legend=dict(orientation="h",yanchor="bottom",y=-.25,xanchor="left",x=0),margin=dict(l=70,r=30,t=80,b=150))
    fig.update_yaxes(range=[np.log10(p_from_si(pmin_used,units)),np.log10(p_from_si(pmax_used,units))])
    st.plotly_chart(fig,use_container_width=True,config={"scrollZoom":True,"displaylogo":False,"toImageButtonOptions":{"format":"png","filename":f"ThermoTools_{FLUID}_PH_Diagram","scale":2}})
    if st.session_state.ph_points:
        rows=[]
        for z in st.session_state.ph_points:
            rows.append({"Point":z["name"],PUNIT(units):p_from_si(z["P"],units),TUNIT(units):t_from_si(z["T"],units),HUNIT(units):h_from_si(z["H"],units),"Density":rho_from_si(z["D"],units),"Phase":z["phase"]})
        df=pd.DataFrame(rows); st.subheader("State point summary"); st.dataframe(df,use_container_width=True,hide_index=True)
        st.download_button("Download state point data (CSV)",df.to_csv(index=False),file_name=f"thermotools_{FLUID.lower()}_state_points.csv",mime="text/csv")
    c1,c2=st.columns(2)
    with c1:
        if st.button("Clear process paths"): st.session_state.ph_paths=[]; st.rerun()
    with c2:
        if st.button("Clear all state points and paths"): st.session_state.ph_points=[]; st.session_state.ph_paths=[]; st.rerun()
    st.caption("Saturation and quality curves shown here are for pure-fluid equilibrium. Mixtures will need a separate phase-envelope calculation.")
    st.warning("CoolProp is a property model, not independent plant validation. Verify model range and accuracy before relying on high-pressure or near-critical results.")

# -------------------- STATE POINT CALCULATOR --------------------
elif page == "State Point Calculator":
    st.title(f"{FLUID} State Point Calculator")
    c1,c2=st.columns(2)
    with c1: pdsp=st.number_input(PLABEL(units),min_value=.001,value=100.0 if units=="Imperial" else .6895)
    with c2: tdsp=st.number_input(TLABEL(units),value=100.0 if units=="Imperial" else 37.78)
    show_model_warnings(FLUID, units, [("Requested state", p_to_si(pdsp, units), t_to_si(tdsp, units))])
    if st.button("Calculate properties"):
        try:
            p,t=p_to_si(pdsp,units),(f_to_k(tdsp) if units=="Imperial" else tdsp+273.15)
            z=state_pt(p,t); st.subheader("Calculated properties")
            vals=[("Enthalpy",h_from_si(z["H"],units),HUNIT(units)),("Entropy",s_from_si(z["S"],units),"BTU/lb-R" if units=="Imperial" else "kJ/kg-K"),("Density",rho_from_si(z["D"],units),"lb/ft³" if units=="Imperial" else "kg/m³"),("Internal energy",h_from_si(z["U"],units),HUNIT(units)),("Cp",cp_from_si(z["CP"],units) if z["CP"] is not None else None,"BTU/lb-R" if units=="Imperial" else "kJ/kg-K"),("Cv",cp_from_si(z["CV"],units) if z["CV"] is not None else None,"BTU/lb-R" if units=="Imperial" else "kJ/kg-K"),("Compressibility factor",z["Z"],"-")]
            cols=st.columns(3)
            for i,(label,value,unit) in enumerate(vals):
                with cols[i%3]: st.metric(label,"N/A" if value is None else f"{value:.5f} {unit}")
            st.subheader("Phase"); st.write(z["phase"])
            pc,tc=PropsSI("Pcrit",FLUID),PropsSI("Tcrit",FLUID)
            st.caption(f"Critical point: {t_from_si(tc,units):.3f} {TUNIT(units)} and {p_from_si(pc,units):.3f} {PUNIT(units)}")
        except Exception as e: explain_coolprop_error(e, FLUID, units)

# -------------------- ISENTHALPIC EXPANSION --------------------
elif page == "Isenthalpic Expansion":
    st.title(f"{FLUID} Isenthalpic Expansion Calculator")
    st.write("Calculate the final temperature following a pressure reduction at constant enthalpy.")
    c1,c2=st.columns(2)
    with c1:
        p1d=st.number_input(f"Initial pressure ({PUNIT(units)})",min_value=.001,value=500.0 if units=="Imperial" else 3.447)
        t1d=st.number_input(f"Initial temperature ({TUNIT(units)})",value=100.0 if units=="Imperial" else 37.78)
    with c2: p2d=st.number_input(f"Final pressure ({PUNIT(units)})",min_value=.001,value=100.0 if units=="Imperial" else .6895)
    show_model_warnings(FLUID, units, [
        ("Initial state", p_to_si(p1d, units), t_to_si(t1d, units)),
        ("Final pressure", p_to_si(p2d, units), None),
    ])
    if st.button("Calculate expansion"):
        try:
            p1,p2=p_to_si(p1d,units),p_to_si(p2d,units); t1=f_to_k(t1d) if units=="Imperial" else t1d+273.15
            h1=PropsSI("H","P",p1,"T",t1,FLUID); t2=PropsSI("T","P",p2,"H",h1,FLUID); h2=PropsSI("H","P",p2,"T",t2,FLUID)
            d1,d2=PropsSI("D","P",p1,"T",t1,FLUID),PropsSI("D","P",p2,"T",t2,FLUID)
            delta=t_from_si(t2,units)-t_from_si(t1,units)
            a,b,c=st.columns(3)
            a.metric("Final temperature",f"{t_from_si(t2,units):.3f} {TUNIT(units)}")
            b.metric("Temperature change",f"{delta:.3f} {TUNIT(units)}")
            c.metric("Enthalpy difference",f"{h_from_si(h2-h1,units):.8f} {HUNIT(units)}")
            st.dataframe(pd.DataFrame({"Property":["Pressure","Temperature","Enthalpy","Density"],"Initial state":[p1d,t1d,h_from_si(h1,units),rho_from_si(d1,units)],"Final state":[p2d,t_from_si(t2,units),h_from_si(h2,units),rho_from_si(d2,units)]}),use_container_width=True,hide_index=True)
            st.write(f"Initial phase: {phase_name(p1,t1)} | Final phase: {phase_name(p2,t2)}")
            st.info("Assumptions: adiabatic throttling, constant enthalpy, and negligible kinetic and potential energy changes.")
        except Exception as e: explain_coolprop_error(e, FLUID, units)

# -------------------- JOULE-THOMSON CALCULATOR --------------------
elif page == "Joule-Thomson Calculator":
    st.title(f"{FLUID} Joule-Thomson Calculator")
    st.write("Calculate the local Joule-Thomson coefficient: μJT = (∂T/∂P) at constant enthalpy.")
    c1,c2=st.columns(2)
    with c1: pdsp=st.number_input(PLABEL(units),min_value=.001,value=500.0 if units=="Imperial" else 3.447)
    with c2: tdsp=st.number_input(TLABEL(units),value=100.0 if units=="Imperial" else 37.78)
    show_model_warnings(FLUID, units, [("Requested state", p_to_si(pdsp, units), t_to_si(tdsp, units))])
    if st.button("Calculate Joule-Thomson coefficient"):
        try:
            p,t=p_to_si(pdsp,units),(f_to_k(tdsp) if units=="Imperial" else tdsp+273.15); value=jt(p,t)
            st.metric("Joule-Thomson coefficient",f"{jt_from_si(value,units):.6f} {'°F/psi' if units=='Imperial' else 'K/MPa'}")
            if value>0: st.success("Positive μJT: a small isenthalpic pressure reduction produces cooling.")
            elif value<0: st.warning("Negative μJT: a small isenthalpic pressure reduction produces heating.")
            else: st.info("The local Joule-Thomson coefficient is zero.")
            st.write(f"Phase: {phase_name(p,t)}")
        except Exception as e: explain_coolprop_error(e, FLUID, units)
    st.divider(); st.subheader("Joule-Thomson reference graph")
    a,b,c=st.columns(3)
    with a: pmin_d=st.number_input(f"Minimum pressure ({PUNIT(units)})",min_value=.001,value=100.0 if units=="Imperial" else .6895,key="jt_min")
    with b: pmax_d=st.number_input(f"Maximum pressure ({PUNIT(units)})",min_value=.01,value=5000.0 if units=="Imperial" else 34.47,key="jt_max")
    with c: count=st.slider("Pressure calculation points",20,500,120)
    tmin=st.number_input(f"Minimum curve temperature ({TUNIT(units)})",value=50.0 if units=="Imperial" else 10.0,key="jt_tmin")
    tmax=st.number_input(f"Maximum curve temperature ({TUNIT(units)})",value=500.0 if units=="Imperial" else 250.0,key="jt_tmax")
    step=st.number_input("Temperature spacing",min_value=1.0,value=50.0,key="jt_step")
    if pmax_d<=pmin_d or tmax<=tmin: st.error("Check that maximum pressure and temperature exceed their minimums.")
    else:
        pgrid=np.linspace(p_to_si(pmin_d,units),p_to_si(pmax_d,units),count); graph=go.Figure(); rows=[]
        for td in np.arange(tmin,tmax+step*.5,step):
            tk=f_to_k(td) if units=="Imperial" else td+273.15; xs=[]; ys=[]
            for p in pgrid:
                try:
                    v=jt_from_si(jt(p,tk),units); pp=p_from_si(p,units); xs.append(pp); ys.append(v); rows.append({f"Temperature ({TUNIT(units)})":td,f"Pressure ({PUNIT(units)})":pp,"JT coefficient":v})
                except Exception: xs.append(p_from_si(p,units)); ys.append(None)
            graph.add_trace(go.Scatter(x=xs,y=ys,mode="lines",name=f"{td:g}{TUNIT(units)}",connectgaps=False))
        graph.update_layout(title=f"{FLUID} Joule-Thomson Coefficient",xaxis_title=PLABEL(units),yaxis_title="μJT (°F/psi)" if units=="Imperial" else "μJT (K/MPa)",template="plotly_dark",height=650,hovermode="closest")
        graph.add_hline(y=0,line_dash="dash"); st.plotly_chart(graph,use_container_width=True)
        if rows: st.download_button("Download JT graph data (CSV)",pd.DataFrame(rows).to_csv(index=False),file_name=f"{FLUID.lower()}_jt_data.csv",mime="text/csv")
    st.warning("The Joule-Thomson coefficient is a local derivative, not the temperature change across a large finite pressure reduction. Use the Isenthalpic Expansion calculator for finite expansions.")


# -------------------- EOS COMPARISON --------------------
elif page == "EOS Comparison":
    st.title(f"{PURE_FLUID} — Equation of State Comparison")
    st.write("Compare the available thermodynamic models at the same pressure and temperature. Each model is evaluated independently.")
    c1, c2 = st.columns(2)
    with c1:
        pdsp = st.number_input(PLABEL(units), min_value=0.001, value=500.0 if units == "Imperial" else 3.447, key="eos_compare_p")
    with c2:
        tdsp = st.number_input(TLABEL(units), value=100.0 if units == "Imperial" else 37.78, key="eos_compare_t")
    st.caption("Enthalpy and entropy reference states may differ across backends. Do not interpret differences in their absolute values as physical discrepancies.")
    if st.button("Compare equations of state", type="primary"):
        p_si, t_si = p_to_si(pdsp, units), t_to_si(tdsp, units)
        results = []
        for backend in (["HEOS", "IF97", "PR", "SRK"] if PURE_FLUID == "Water" else ["HEOS", "PR", "SRK"]):
            fluid_id = f"{backend}::{PURE_FLUID}"
            record = {"EOS": backend, "Status": "OK"}
            errors = []
            for label, output, convert in [
                ("Density (lb/ft³)" if units == "Imperial" else "Density (kg/m³)", "D", rho_from_si),
                ("Compressibility Z", "Z", lambda x, u: x),
                (f"Enthalpy ({HUNIT(units)})", "H", h_from_si),
                (f"Entropy ({'BTU/lb-R' if units == 'Imperial' else 'kJ/kg-K'})", "S", s_from_si),
            ]:
                try:
                    record[label] = convert(PropsSI(output, "P", p_si, "T", t_si, fluid_id), units)
                except Exception as exc:
                    record[label] = None
                    errors.append(f"{label}: {exc}")
            if errors:
                record["Status"] = "Partial" if len(errors) < 4 else "Unavailable"
                record["Error"] = "\n".join(errors)
            results.append(record)
        st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
        for record in results:
            if record.get("Error"):
                with st.expander(f"{record['EOS']} — calculation details"):
                    st.code(record["Error"])
        st.warning("Cubic EOS predictions are screening estimates; numerical success does not establish validity at extreme pressure, near phase transitions, or where solids may form.")


# -------------------- HYDROCARBON MIXTURE BUILDER (STAGE 1) --------------------
elif page == "Mixture Builder":
    st.title("Hydrocarbon Mixture Builder")
    st.write("Create a hydrocarbon mixture, enter its composition on a mole or mass basis, and calculate a state point. The pure-fluid P-H diagram and its plotting tools are unchanged.")
    st.info("Mixture calculations use CoolProp's HEOS mixture model. Not every combination or state is supported. Results are screening calculations, not validated design data.")

    hydrocarbon_names = list(dict.fromkeys(
        FLUID_CATEGORIES.get("Light Alkanes", []) +
        FLUID_CATEGORIES.get("Alkenes and Unsaturated Hydrocarbons", []) +
        [x for x in ["Benzene", "Toluene", "CycloHexane"] if x in AVAILABLE]
    ))
    if len(hydrocarbon_names) < 2:
        st.error("The installed CoolProp library does not have enough hydrocarbons for mixture calculations.")
        st.stop()

    basis = st.radio("Composition entry basis", ["Mole %", "Mass %"], horizontal=True)
    count = st.number_input("Number of components", min_value=2, max_value=6, value=2, step=1)
    st.caption("Enter fractions totaling 100%. Component names must be unique.")
    defaults = ["Ethylene", "Ethane", "Propane", "Methane", "Propylene", "n-Butane"]
    components, entered = [], []
    for i in range(int(count)):
        left, right = st.columns([3, 2])
        preferred = defaults[i] if defaults[i] in hydrocarbon_names else hydrocarbon_names[min(i, len(hydrocarbon_names)-1)]
        with left:
            name = st.selectbox(f"Component {i+1}", hydrocarbon_names,
                                index=hydrocarbon_names.index(preferred), key=f"mix_component_{i}")
        with right:
            amount = st.number_input(f"{basis} — component {i+1}", min_value=0.0, max_value=100.0,
                                     value=95.0 if i == 0 else (5.0 if i == 1 else 0.0),
                                     step=0.1, key=f"mix_amount_{i}_{basis}")
        components.append(name)
        entered.append(amount)

    total = sum(entered)
    st.write(f"**Entered total: {total:.4f}%**")
    if len(set(components)) != len(components):
        st.error("Select each component only once.")
        st.stop()
    if abs(total - 100.0) > 0.001:
        st.warning("Composition must total 100% before properties can be calculated.")
        st.stop()

    # Convert all compositions to mole fractions for the HEOS backend.
    try:
        molecular_weights = [PropsSI("M", component) for component in components]  # kg/mol
        fractions = [value / 100.0 for value in entered]
        if basis == "Mass %":
            molar_amounts = [w / mw for w, mw in zip(fractions, molecular_weights)]
            mole_fractions = [n / sum(molar_amounts) for n in molar_amounts]
        else:
            mole_fractions = fractions
        mixture_mw = sum(x * mw for x, mw in zip(mole_fractions, molecular_weights))
        mass_fractions = [x * mw / mixture_mw for x, mw in zip(mole_fractions, molecular_weights)]
    except Exception as exc:
        st.error(f"Unable to calculate composition conversion: {exc}")
        st.stop()

    st.subheader("Calculated composition")
    comp_table = pd.DataFrame({
        "Component": components,
        "Molecular weight (g/mol)": [mw * 1000 for mw in molecular_weights],
        "Mole %": [x * 100 for x in mole_fractions],
        "Mass %": [w * 100 for w in mass_fractions],
    })
    st.dataframe(comp_table.style.format({"Molecular weight (g/mol)": "{:.4f}",
                                        "Mole %": "{:.5f}", "Mass %": "{:.5f}"}),
                 use_container_width=True, hide_index=True)
    st.metric("Mixture molecular weight", f"{mixture_mw * 1000:.4f} g/mol")
    st.download_button("Download composition (CSV)", comp_table.to_csv(index=False),
                       file_name="thermotools_mixture_composition.csv", mime="text/csv")

    # CoolProp uses mole fractions for mixture strings. Components with zero fraction
    # are omitted, avoiding unsupported zero-concentration endpoints.
    active = [(name, x) for name, x in zip(components, mole_fractions) if x > 1e-12]
    active_total = sum(x for _, x in active)
    fluid_string = "HEOS::" + "&".join(f"{name}[{x/active_total:.12g}]" for name, x in active)

    st.session_state.mixture_definition = {"components": components, "mole_fractions": mole_fractions, "fluid_string": fluid_string, "mw": mixture_mw}
    st.subheader("Mixture state-point calculator")
    left, right = st.columns(2)
    with left:
        pressure = st.number_input(f"Pressure ({PUNIT(units)})", min_value=0.001,
                                   value=100.0 if units == "Imperial" else 0.6895,
                                   key="mix_pressure")
    with right:
        temperature = st.number_input(f"Temperature ({TUNIT(units)})",
                                      value=100.0 if units == "Imperial" else 37.78,
                                      key="mix_temperature")
    if st.button("Calculate mixture state", type="primary"):
        p_si = p_to_si(pressure, units)
        t_si = t_to_si(temperature, units)
        try:
            enthalpy = PropsSI("H", "P", p_si, "T", t_si, fluid_string)
            entropy = PropsSI("S", "P", p_si, "T", t_si, fluid_string)
            density = PropsSI("D", "P", p_si, "T", t_si, fluid_string)
            internal_energy = PropsSI("U", "P", p_si, "T", t_si, fluid_string)
            values = [
                ("Enthalpy", h_from_si(enthalpy, units), HUNIT(units)),
                ("Entropy", s_from_si(entropy, units), "BTU/lb-R" if units == "Imperial" else "kJ/kg-K"),
                ("Density", rho_from_si(density, units), "lb/ft³" if units == "Imperial" else "kg/m³"),
                ("Internal energy", h_from_si(internal_energy, units), HUNIT(units)),
            ]
            for label, output, conversion, unit in [
                ("Cp", "C", cp_from_si, "BTU/lb-R" if units == "Imperial" else "kJ/kg-K"),
                ("Cv", "O", cp_from_si, "BTU/lb-R" if units == "Imperial" else "kJ/kg-K"),
                ("Compressibility factor", "Z", lambda x, u: x, ""),
            ]:
                try:
                    val = PropsSI(output, "P", p_si, "T", t_si, fluid_string)
                    values.append((label, conversion(val, units), unit))
                except Exception:
                    values.append((label, None, unit))
            columns = st.columns(3)
            for i, (label, val, unit) in enumerate(values):
                with columns[i % 3]:
                    st.metric(label, "N/A" if val is None else f"{val:.5f} {unit}")
            try:
                phase = PhaseSI("P", p_si, "T", t_si, fluid_string)
                st.write(f"**CoolProp phase classification:** {phase.replace('_', ' ').title()}")
            except Exception:
                st.caption("Phase classification is not available for this mixture and state.")
            st.success("Mixture state calculated successfully.")
        except Exception as exc:
            st.error("CoolProp could not evaluate this mixture at the selected pressure and temperature.")
            st.code(str(exc))
            st.caption("Some hydrocarbon pairs lack binary interaction parameters, and HEOS mixture flash calculations have limited two-phase support. Try another composition or single-phase state.")

    st.success("Mixture saved for the Mixture P-H Diagram page. Select that page from the sidebar to plot it.")

# -------------------- MIXTURE P-H DIAGRAM --------------------
elif page == "Mixture P-H Diagram":
    st.title("Hydrocarbon Mixture Pressure-Enthalpy Diagram")
    mixture = st.session_state.get("mixture_definition")
    if not mixture:
        st.info("First open Mixture Builder, enter a composition totaling 100%, then return here.")
        st.stop()
    mix_fluid = mixture["fluid_string"]
    mix_label = " + ".join(f"{name} {x*100:.3g}%" for name, x in zip(mixture["components"], mixture["mole_fractions"]) if x > 1e-12)
    st.write(f"**Current composition (mole basis):** {mix_label}")
    st.caption("Fixed overall composition. Both diagrams use the same CoolProp phase-envelope calculation; bubble and dew branches are identified by their equilibrium endpoints.")
    c1, c2, c3 = st.columns(3)
    with c1:
        low = st.number_input(f"Minimum pressure ({PUNIT(units)})", min_value=0.001, value=5.0 if units == "Imperial" else 0.035, key="mix_ph_low")
    with c2:
        high = st.number_input(f"Maximum pressure ({PUNIT(units)})", min_value=0.002, value=1000.0 if units == "Imperial" else 7.0, key="mix_ph_high")
    with c3:
        samples = st.select_slider("Resolution", options=[20, 40, 70], value=40, key="mix_ph_samples")
    if low >= high:
        st.error("Maximum pressure must exceed minimum pressure.")
        st.stop()
    pgrid = np.geomspace(p_to_si(low, units), p_to_si(high, units), samples)
    show_isotherms = st.checkbox("Show mixture isotherms", value=False, key="mix_ph_isotherms")
    fig = go.Figure()
    tp_fig = go.Figure()

    # Build the *whole* mixture phase envelope in one continuation calculation.
    # Independent P,Q flashes can jump between unrelated branches near criticality.
    @st.cache_data(show_spinner=False)
    def mixture_envelope(component_tuple, fraction_tuple):
        state = AbstractState("HEOS", "&".join(component_tuple))
        state.set_mole_fractions(list(fraction_tuple))
        state.build_phase_envelope("")
        envelope = state.get_phase_envelope_data()
        mw = sum(PropsSI("M", c) * x for c, x in zip(component_tuple, fraction_tuple))
        records = []
        for p, temp, quality, hl, hv in zip(envelope.p, envelope.T, envelope.Q,
                                             envelope.hmolar_liq, envelope.hmolar_vap):
            if not all(np.isfinite(v) for v in (p, temp, quality, hl, hv)):
                continue
            # Q=0: liquid endpoint; Q=1: vapor endpoint.
            if abs(quality) < 0.01:
                branch, hmass = "Bubble point (Q=0)", hl / mw
            elif abs(quality - 1) < 0.01:
                branch, hmass = "Dew point (Q=1)", hv / mw
            else:
                continue
            records.append((branch, float(p), float(temp), float(hmass)))
        return records

    counts = {"Bubble point (Q=0)": 0, "Dew point (Q=1)": 0}
    envelope_ok = False
    try:
        active_pairs = [(c, x) for c, x in zip(mixture["components"], mixture["mole_fractions"]) if x > 1e-12]
        active_sum = sum(x for _, x in active_pairs)
        component_tuple = tuple(c for c, _ in active_pairs)
        fraction_tuple = tuple(x / active_sum for _, x in active_pairs)
        with st.spinner("Calculating mixture phase envelope..."):
            envelope_records = mixture_envelope(component_tuple, fraction_tuple)
        envelope_ok = True
        # Preserve the phase-envelope solver's ordering and separate the two branches.
        # Do not bridge missing pressure regions or nonphysical enthalpy jumps.
        for branch, color in [("Bubble point (Q=0)", "#21c6b7"),
                              ("Dew point (Q=1)", "#ffa65c")]:
            branch_records = [(p, t, h) for b, p, t, h in envelope_records
                              if b == branch and p_to_si(low, units) <= p <= p_to_si(high, units)]
            counts[branch] = len(branch_records)
            if not branch_records:
                continue
            xs, ys, hovers, tx = [], [], [], []
            last = None
            for p, t, h in branch_records:
                # Split curves at large discontinuities instead of connecting them.
                if last is not None:
                    oldp, oldt, oldh = last
                    if (max(p, oldp) / min(p, oldp) > 2.0 or
                        abs(t - oldt) > 90 or abs(h - oldh) > 350000):
                        xs.append(None); ys.append(None); hovers.append(""); tx.append(None)
                xs.append(h_from_si(h, units))
                tx.append(t_from_si(t, units))
                ys.append(p_from_si(p, units))
                hovers.append(f"T: {t_from_si(t, units):.2f} {TUNIT(units)}")
                last = (p, t, h)
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=branch,
                                     line=dict(color=color, width=3), marker=dict(size=4),
                                     text=hovers, connectgaps=False,
                                     hovertemplate="H: %{x:.2f}<br>P: %{y:.2f}<br>%{text}<extra>%{fullData.name}</extra>"))
            tp_fig.add_trace(go.Scatter(x=tx, y=ys, mode="lines+markers", name=branch,
                                        line=dict(color=color, width=3), marker=dict(size=4),
                                        connectgaps=False,
                                        hovertemplate="T: %{x:.2f}<br>P: %{y:.2f}<extra>%{fullData.name}</extra>"))
    except Exception as exc:
        st.warning("CoolProp could not build a continuous phase envelope for this composition. "
                   "No bubble/dew curves are drawn rather than showing potentially incorrect curves.")
        with st.expander("Phase-envelope calculation details"):
            st.code(str(exc))

    if show_isotherms:
        temperature_choices = [-100, 0, 100, 200, 300] if units == "Imperial" else [-75, -25, 25, 75, 125]
        for temp in temperature_choices:
            temp_si = t_to_si(temp, units)
            hx, py = [], []
            for pv in pgrid:
                try:
                    hv = PropsSI("H", "P", float(pv), "T", float(temp_si), mix_fluid)
                    hx.append(h_from_si(hv, units) if np.isfinite(hv) else None)
                except Exception:
                    hx.append(None)
                py.append(p_from_si(pv, units))
            fig.add_trace(go.Scatter(x=hx, y=py, mode="lines", name=f"{temp} {TUNIT(units)}",
                                     line=dict(width=1, dash="dot"), connectgaps=False))
    st.subheader("Add mixture state points")
    mixture_id = mix_fluid
    if st.session_state.get("mix_ph_identity") != mixture_id:
        st.session_state.mix_ph_points = []
        st.session_state.mix_ph_paths = []
        st.session_state.mix_ph_identity = mixture_id
    with st.form("mix_ph_point_form"):
        ncol, pcol, tcol = st.columns(3)
        with ncol: point_name = st.text_input("Point name", value=f"Point {len(st.session_state.mix_ph_points)+1}")
        with pcol: point_p = st.number_input(f"Pressure ({PUNIT(units)})", min_value=0.001, value=100.0 if units == "Imperial" else 0.7)
        with tcol: point_t = st.number_input(f"Temperature ({TUNIT(units)})", value=100.0 if units == "Imperial" else 37.8)
        submitted = st.form_submit_button("Calculate and add point")
    if submitted:
        try:
            p_si, t_si = p_to_si(point_p, units), t_to_si(point_t, units)
            h_si = PropsSI("H", "P", p_si, "T", t_si, mix_fluid)
            if not np.isfinite(h_si): raise ValueError("Nonfinite enthalpy")
            name = point_name.strip() or f"Point {len(st.session_state.mix_ph_points)+1}"
            if any(z["name"] == name for z in st.session_state.mix_ph_points): raise ValueError("Point names must be unique")
            st.session_state.mix_ph_points.append(dict(name=name, P=p_si, T=t_si, H=h_si))
            st.success(f"Added {name}.")
        except Exception as exc:
            st.error(f"Could not calculate mixture state: {exc}")
    if st.session_state.mix_ph_points:
        st.subheader("Calculate isenthalpic expansion")
        point_names = [z["name"] for z in st.session_state.mix_ph_points]
        selected = st.selectbox("Starting point", point_names, key="mix_ph_start")
        start = next(z for z in st.session_state.mix_ph_points if z["name"] == selected)
        target = st.number_input(f"Final pressure ({PUNIT(units)})", min_value=0.001, value=float(max(0.001, p_from_si(start["P"], units)*0.5)), key="mix_ph_target")
        if st.button("Calculate and plot isenthalpic expansion"):
            p2 = p_to_si(target, units)
            if p2 >= start["P"]:
                st.error("Final pressure must be below starting pressure.")
            else:
                path_p = np.geomspace(start["P"], p2, 45)
                valid_p, valid_h = [], []
                try:
                    # Validate the P,H flash at each pressure, not just the endpoints.
                    for pp in path_p:
                        tcheck = PropsSI("T", "P", float(pp), "H", start["H"], mix_fluid)
                        if not np.isfinite(tcheck): raise ValueError("Nonfinite flash result")
                        valid_p.append(float(pp)); valid_h.append(start["H"])
                    t2 = PropsSI("T", "P", p2, "H", start["H"], mix_fluid)
                    st.session_state.mix_ph_paths.append(dict(name=f"{selected}: isenthalpic to {target:g} {PUNIT(units)}", p=valid_p, h=valid_h))
                    st.success(f"Calculated final temperature: {t_from_si(t2, units):.3f} {TUNIT(units)}")
                except Exception as exc:
                    st.error(f"Mixture P-H flash unavailable along this path: {exc}")
    for path in st.session_state.mix_ph_paths:
        fig.add_trace(go.Scatter(x=[h_from_si(v, units) for v in path["h"]], y=[p_from_si(v, units) for v in path["p"]], mode="lines", name=path["name"], line=dict(width=3)))
    if st.session_state.mix_ph_points:
        fig.add_trace(go.Scatter(x=[h_from_si(z["H"], units) for z in st.session_state.mix_ph_points], y=[p_from_si(z["P"], units) for z in st.session_state.mix_ph_points], mode="markers+text", text=[z["name"] for z in st.session_state.mix_ph_points], textposition="top right", name="State points", marker=dict(size=11, symbol="diamond")))
        tp_fig.add_trace(go.Scatter(x=[t_from_si(z["T"], units) for z in st.session_state.mix_ph_points], y=[p_from_si(z["P"], units) for z in st.session_state.mix_ph_points], mode="markers+text", text=[z["name"] for z in st.session_state.mix_ph_points], textposition="top right", name="State points", marker=dict(size=11, symbol="diamond")))
    fig.update_layout(title="Mixture Pressure-Enthalpy Diagram", xaxis_title=HLABEL(units), yaxis_title=PLABEL(units), yaxis_type="log", template="plotly_dark", height=750, hovermode="closest")
    fig.update_yaxes(range=[np.log10(low), np.log10(high)])
    tp_fig.update_layout(title="Mixture Temperature-Pressure Diagram", xaxis_title=TLABEL(units),
                         yaxis_title=PLABEL(units), yaxis_type="log", template="plotly_dark",
                         height=750, hovermode="closest")
    tp_fig.update_yaxes(range=[np.log10(low), np.log10(high)])
    left_chart, right_chart = st.columns(2)
    with left_chart:
        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "displaylogo": False, "toImageButtonOptions": {"format": "png", "filename": "ThermoTools_mixture_PH", "scale": 2}})
    with right_chart:
        st.plotly_chart(tp_fig, use_container_width=True, config={"scrollZoom": True, "displaylogo": False, "toImageButtonOptions": {"format": "png", "filename": "ThermoTools_mixture_TP", "scale": 2}})
    st.caption("The P-H chart includes optional isotherms; the T-P chart shows the phase envelope and saved state points. Isenthalpic paths are displayed only on P-H until their intermediate temperatures can be validated.")
    st.caption(f"Phase-envelope points inside selected pressure range: bubble {counts['Bubble point (Q=0)']}; dew {counts['Dew point (Q=1)']}. Curves are calculated for the fixed overall mixture composition.")
    if envelope_ok and min(counts.values()) < 2:
        st.warning("Too few phase-envelope points fall inside the selected pressure range. Try widening the pressure range; the envelope may lie partly outside it.")
    if st.session_state.mix_ph_points:
        rows = [{"Point": z["name"], PUNIT(units): p_from_si(z["P"], units), TUNIT(units): t_from_si(z["T"], units), HUNIT(units): h_from_si(z["H"], units)} for z in st.session_state.mix_ph_points]
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("Download mixture state points (CSV)", df.to_csv(index=False), file_name="mixture_state_points.csv", mime="text/csv")
    if st.button("Clear mixture points and paths"):
        st.session_state.mix_ph_points = []
        st.session_state.mix_ph_paths = []
        st.rerun()
    st.warning("Mixture phase equilibrium and near-critical calculations can be numerically fragile. Curves are screening results, not validated process-design data. Do not interpolate across missing curve segments.")
