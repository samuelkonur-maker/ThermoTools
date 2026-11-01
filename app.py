import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from CoolProp.CoolProp import PropsSI, PhaseSI, get_global_param_string, AbstractState

st.set_page_config(page_title="ThermoTools", page_icon="🌡️", layout="wide")

# -------------------- CHEMICAL LIBRARY --------------------
FLUID_CATEGORIES = {
    "Light Alkanes": ["Methane", "Ethane", "Propane", "n-Butane", "IsoButane", "n-Pentane", "IsoPentane", "n-Hexane", "n-Heptane", "n-Octane", "n-Decane"],
    "Alkenes and Unsaturated Hydrocarbons": ["Ethylene", "Propylene", "1-Butene", "cis-2-Butene", "trans-2-Butene", "IsoButylene", "1-Hexene", "1-Octene", "Acetylene", "1,3-Butadiene"],
    "Process Gases": ["Hydrogen", "Nitrogen", "Oxygen", "CarbonDioxide", "CarbonMonoxide", "Ammonia"],
    "Water": ["Water"],
    "Aromatics and Other Hydrocarbons": ["Benzene", "Toluene", "EthylBenzene", "Styrene", "CycloHexane"],
    "Solvents and Refrigerants": ["Methanol", "Ethanol", "Acetone", "R134a"]
}
AVAILABLE = set(get_global_param_string("FluidsList").split(","))
FLUID_CATEGORIES = {k: [f for f in v if f in AVAILABLE] for k, v in FLUID_CATEGORIES.items()}
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
def get_pure_fluid_limits(fluid):
    """Query the selected HEOS model; unavailable limits remain None."""
    result = {"Tmin": None, "Tmax": None, "pmax": None, "pmin": None,
              "Tcrit": None, "pcrit": None}
    try:
        model = AbstractState("HEOS", fluid)
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
    limits = get_pure_fluid_limits(fluid.split("::")[-1])
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
        st.warning(("HEOS reference limits (not PR/SRK validation): " if EOS != "HEOS" else "CoolProp model range: ") + message)


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
page = st.sidebar.radio("Navigation", ["Home", "Pressure-Enthalpy Diagram", "State Point Calculator", "Isenthalpic Expansion", "Joule-Thomson Calculator", "Mixture Builder", "Mixture P-H Diagram", "EOS Comparison"])
units = st.sidebar.radio("Unit System", ["Imperial", "SI"], horizontal=True)
st.sidebar.divider()
st.sidebar.subheader("Fluid Selection")
category = st.sidebar.selectbox("Chemical category", list(FLUID_CATEGORIES))
fluids = FLUID_CATEGORIES[category]
default_idx = fluids.index("Ethylene") if "Ethylene" in fluids else 0
PURE_FLUID = st.sidebar.selectbox("Chemical", fluids, index=default_idx)
EOS = st.sidebar.selectbox("Equation of state", ["HEOS", "PR", "SRK"],
                           format_func=lambda x: {"HEOS":"Helmholtz (HEOS)", "PR":"Peng–Robinson (PR)", "SRK":"Soave–Redlich–Kwong (SRK)"}[x])
FLUID = f"{EOS}::{PURE_FLUID}"
st.sidebar.caption("Mixture tools currently use HEOS regardless of this selection.")
old_fluid = st.session_state.get("active_fluid")
if old_fluid is not None and old_fluid != FLUID:
    st.session_state.ph_points = []
    st.session_state.ph_paths = []
    st.sidebar.info("Fluid changed; saved state points and paths were cleared.")
st.session_state.active_fluid = FLUID
st.sidebar.caption(f"Active fluid: {FLUID}")
st.sidebar.divider()
with st.sidebar.expander("CoolProp model limits", expanded=True):
    st.caption(f"Selected backend: {EOS} / {PURE_FLUID}")
    _limits = get_pure_fluid_limits(PURE_FLUID)
    if EOS != "HEOS":
        st.info("The limits below are from the HEOS fluid data, not validated PR/SRK operating limits. Cubic EOS flash support can be narrower.")
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
    st.write("Compare HEOS, Peng–Robinson, and SRK at the same pressure and temperature. Each model is evaluated independently.")
    c1, c2 = st.columns(2)
    with c1:
        pdsp = st.number_input(PLABEL(units), min_value=0.001, value=500.0 if units == "Imperial" else 3.447, key="eos_compare_p")
    with c2:
        tdsp = st.number_input(TLABEL(units), value=100.0 if units == "Imperial" else 37.78, key="eos_compare_t")
    st.caption("Enthalpy and entropy reference states may differ across backends. Do not interpret differences in their absolute values as physical discrepancies.")
    if st.button("Compare equations of state", type="primary"):
        p_si, t_si = p_to_si(pdsp, units), t_to_si(tdsp, units)
        results = []
        for backend in ("HEOS", "PR", "SRK"):
            fluid_id = f"{backend}::{PURE_FLUID}"
            record = {"EOS": backend, "Status": "OK"}
            try:
                for label, output, convert in [
                    ("Density (lb/ft³)" if units == "Imperial" else "Density (kg/m³)", "D", rho_from_si),
                    ("Compressibility Z", "Z", lambda x, u: x),
                    (f"Enthalpy ({HUNIT(units)})", "H", h_from_si),
                    (f"Entropy ({'BTU/lb-R' if units == 'Imperial' else 'kJ/kg-K'})", "S", s_from_si),
                ]:
                    record[label] = convert(PropsSI(output, "P", p_si, "T", t_si, fluid_id), units)
            except Exception as exc:
                record["Status"] = "Unavailable"
                record["Error"] = str(exc)
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
