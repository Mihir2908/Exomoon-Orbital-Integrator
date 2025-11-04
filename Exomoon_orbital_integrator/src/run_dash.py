import os, sys
from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objects as go

# Ensure src/ is importable regardless of current working dir
SRC_DIR = os.path.dirname(__file__)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from exomoon.params import SystemParams
from exomoon.simulation import run_simulation
from exomoon.plotting.anim import build_animation

app = Dash(__name__)
app.title = "Exomoon Orbital Integrator (Interactive)"

# Use current defaults from params.py
_defaults = SystemParams()

controls = html.Div([
    html.H4("Inputs"),
    html.Label("Stellar temperature Ts (K)"),
    dcc.Input(id="Ts", type="number", value=_defaults.Ts, min=2500, max=8000, step="any", style={"width": "100%"}),

    html.Label("Star radius (R☉)"),
    dcc.Slider(id="rs_solar", min=0.1, max=2.0, step=0.01, value=_defaults.rs_solar, tooltip={"placement": "bottom"}),

    html.Label("Star mass (M☉)"),
    dcc.Slider(id="ms_solar", min=0.1, max=2.0, step=0.01, value=_defaults.ms_solar, tooltip={"placement": "bottom"}),

    html.Label("Planet mass (M⊕)"),
    dcc.Slider(id="mp_earth", min=0.1, max=10.0, step=0.01, value=_defaults.mp_earth, tooltip={"placement": "bottom"}),

    html.Label("Planet density (g/cc)"),
    dcc.Slider(id="dp_cgs", min=0.5, max=15.0, step=0.1, value=_defaults.dp_cgs, tooltip={"placement": "bottom"}),

    html.Label("Planet semi-major axis ap (AU)"),
    dcc.Slider(id="ap_AU", min=0.05, max=2.0, step=0.001, value=_defaults.ap_AU, tooltip={"placement": "bottom"}),

    html.Label("Planet eccentricity ep"),
    dcc.Slider(id="ep", min=0.0, max=0.8, step=0.01, value=_defaults.ep, tooltip={"placement": "bottom"}),

    html.Label("Moon mass (M⊕)"),
    dcc.Slider(id="mm_earth", min=0.001, max=10.0, step=0.01, value=_defaults.mm_earth, tooltip={"placement": "bottom"}),

    html.Label("Moon a as fraction of Hill radius (am_hill)"),
    dcc.Slider(id="am_hill", min=0.05, max=0.95, step=0.01, value=_defaults.am_hill, tooltip={"placement": "bottom"}),

    html.Label("Moon eccentricity em"),
    dcc.Slider(id="em", min=0.0, max=0.8, step=0.01, value=_defaults.em, tooltip={"placement": "bottom"}),

    html.Button("Run Simulation", id="run-btn", n_clicks=0, style={"width": "100%", "marginTop": "10px"}),
], style={"display": "flex", "flexDirection": "column", "gap": "10px"})

app.layout = html.Div([
    html.Div(controls, style={"width": "340px", "padding": "12px", "borderRight": "1px solid #ddd"}),
    html.Div([
        dcc.Loading(
            id="loading",
            type="default",
            children=dcc.Graph(id="orbit-graph", figure=go.Figure(), style={"height": "90vh"})
        )
    ], style={"flex": "1", "padding": "12px"})
], style={"display": "flex", "height": "100vh", "fontFamily": "Segoe UI, Arial"})

def _fnum(v, default):
    try:
        return default if v is None or (isinstance(v, str) and v.strip() == "") else float(v)
    except Exception:
        return default

@app.callback(
    Output("orbit-graph", "figure"),
    Input("run-btn", "n_clicks"),
    State("Ts", "value"), State("rs_solar", "value"), State("ms_solar", "value"),
    State("mp_earth", "value"), State("dp_cgs", "value"),
    State("ap_AU", "value"), State("ep", "value"),
    State("mm_earth", "value"), State("am_hill", "value"), State("em", "value"),
    prevent_initial_call=False,
)
def run_cb(n_clicks, Ts, rs_solar, ms_solar, mp_earth, dp_cgs, ap_AU, ep, mm_earth, am_hill, em):
    d = SystemParams()  # current defaults as fallback
    p = SystemParams(
        Ts=_fnum(Ts, d.Ts), rs_solar=_fnum(rs_solar, d.rs_solar), ms_solar=_fnum(ms_solar, d.ms_solar),
        mp_earth=_fnum(mp_earth, d.mp_earth), dp_cgs=_fnum(dp_cgs, d.dp_cgs),
        ap_AU=_fnum(ap_AU, d.ap_AU), ep=_fnum(ep, d.ep),
        mm_earth=_fnum(mm_earth, d.mm_earth), am_hill=_fnum(am_hill, d.am_hill), em=_fnum(em, d.em),
    )

    sim = run_simulation(p)
    fig = build_animation(sim["traj"], sim["a_inner_au"], sim["a_outer_au"], open_in_browser=False)

    # Show Hill radius computed in initial_conditions (consistent with the integrator units)
    st = sim.get("state", {})
    rhill = st.get("rhill_AU")
    if isinstance(rhill, (int, float)):
        fig.add_annotation(
            text=f"Hill radius: {rhill:.4f} AU (a_moon ≈ {p.am_hill*rhill:.4f} AU)",
            xref="paper", yref="paper", x=0.01, y=1.06,
            showarrow=False, align="left", font=dict(size=12)
        )

    return fig

if __name__ == "__main__":
    app.run(debug=True)