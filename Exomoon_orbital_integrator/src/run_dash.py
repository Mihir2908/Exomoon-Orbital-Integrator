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
from exomoon.exoplanet_archive import fetch_system_by_planet, estimate_density_gcc

app = Dash(__name__)
app.title = "Exomoon Orbital Integrator (Interactive)"

_defaults = SystemParams()

controls = html.Div([
    html.H4("NASA Exoplanet Archive"),
    html.Div([
        dcc.Input(id="pl_name", type="text", placeholder="Planet name, e.g., HD 209458 b", style={"width": "100%"}),
        html.Button("Fetch from NASA", id="fetch-btn", n_clicks=0, style={"width": "100%", "marginTop": "6px"}),
        html.Div(id="fetch-status", style={"fontSize": "12px", "color": "#666", "marginTop": "4px"}),
    ], style={"display": "flex", "flexDirection": "column", "gap": "4px", "marginBottom": "12px"}),

    html.H4("Inputs"),
    html.Label("Stellar temperature Ts (K)"),
    dcc.Input(id="Ts", type="number", value=_defaults.Ts, min=2000, max=20000, step="any", style={"width": "100%"}),

    html.Label("Star radius (R☉)"),
    dcc.Slider(id="rs_solar", min=0.05, max=30.0, step=0.01, value=_defaults.rs_solar, tooltip={"placement": "bottom"}),

    html.Label("Star mass (M☉)"),
    dcc.Slider(id="ms_solar", min=0.05, max=50.0, step=0.01, value=_defaults.ms_solar, tooltip={"placement": "bottom"}),

    # IMPORTANT: make wide-range sliders continuous (no discrete stepping)
    html.Label("Planet mass (M⊕)"),
    dcc.Slider(id="mp_earth", min=0.01, max=10.0, step=None, value=_defaults.mp_earth, tooltip={"placement": "bottom"}),

    html.Label("Planet density (g/cc)"),
    dcc.Slider(id="dp_cgs", min=0.2, max=40.0, step=0.01, value=_defaults.dp_cgs, tooltip={"placement": "bottom"}),

    html.Label("Planet semi-major axis ap (AU)"),
    dcc.Slider(id="ap_AU", min=0.005, max=10.0, step=None, value=_defaults.ap_AU, tooltip={"placement": "bottom"}),

    html.Label("Planet eccentricity ep"),
    dcc.Slider(id="ep", min=0.0, max=0.99, step=0.001, value=_defaults.ep, tooltip={"placement": "bottom"}),

    html.Label("Moon mass (M⊕)"),
    dcc.Slider(id="mm_earth", min=0.01, max=10.0, step=None, value=_defaults.mm_earth, tooltip={"placement": "bottom"}),

    html.Label("Moon a as fraction of Hill radius (am_hill)"),
    dcc.Slider(id="am_hill", min=0.01, max=0.95, step=0.01, value=_defaults.am_hill, tooltip={"placement": "bottom"}),

    html.Label("Moon eccentricity em"),
    dcc.Slider(id="em", min=0.0, max=1.0, step=0.001, value=_defaults.em, tooltip={"placement": "bottom"}),

    html.Button("Run Simulation", id="run-btn", n_clicks=0, style={"width": "100%", "marginTop": "10px"}),
], style={"display": "flex", "flexDirection": "column", "gap": "10px"})

def _initial_figure():
    fig = go.Figure()
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[dict(
            text="Set parameters or fetch from NASA, then click “Run Simulation”.",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(size=14, color="#555"), align="center"
        )],
        margin=dict(l=20, r=20, t=40, b=20),
        height=700,
        title="Exomoon Orbital Integrator"
    )
    return fig

app.layout = html.Div([
    html.Div(controls, style={"width": "380px", "padding": "12px", "borderRight": "1px solid #ddd"}),
    html.Div([
        dcc.Loading(
            id="loading",
            type="default",
            children=dcc.Graph(id="orbit-graph", figure=_initial_figure(), style={"height": "90vh"})
        )
    ], style={"flex": "1", "padding": "12px"})
], style={"display": "flex", "height": "100vh", "fontFamily": "Segoe UI, Arial"})

def _fnum(v, default):
    try:
        return default if v is None or (isinstance(v, str) and v.strip() == "") else float(v)
    except Exception:
        return default

# Fetch from NASA and fill sliders
@app.callback(
    [
        Output("fetch-status", "children"),
        Output("Ts", "value"),
        Output("rs_solar", "value"),
        Output("ms_solar", "value"),
        Output("mp_earth", "value"),
        Output("ap_AU", "value"),
        Output("ep", "value"),
        Output("dp_cgs", "value"),
    ],
    Input("fetch-btn", "n_clicks"),
    State("pl_name", "value"),
    prevent_initial_call=True,
)
def fetch_from_nasa(n_clicks, pl_name):
    d = SystemParams()
    if not pl_name or not str(pl_name).strip():
        return ("Enter a planet name (e.g., HD 209458 b).", d.Ts, d.rs_solar, d.ms_solar, d.mp_earth, d.ap_AU, d.ep, d.dp_cgs)
    try:
        rec = fetch_system_by_planet(pl_name)
        if not rec:
            return (f"No results for '{pl_name}'.", d.Ts, d.rs_solar, d.ms_solar, d.mp_earth, d.ap_AU, d.ep, d.dp_cgs)

        Ts = rec.get("Ts") or d.Ts
        rs = rec.get("rs_solar") or d.rs_solar
        ms = rec.get("ms_solar") or d.ms_solar
        mp = rec.get("mp_earth") or d.mp_earth
        ap = rec.get("ap_AU") or d.ap_AU
        ep = rec.get("ep") or d.ep

        # Density estimate if both mass and radius are available
        est_rho = estimate_density_gcc(rec.get("mp_earth"), rec.get("pl_rade"))
        dp = est_rho or d.dp_cgs

        status = f"Loaded: {rec.get('pl_name')} (host: {rec.get('hostname')})"
        return (status, Ts, rs, ms, mp, ap, ep, dp)
    except Exception as e:
        return (f"Error fetching '{pl_name}': {e}", d.Ts, d.rs_solar, d.ms_solar, d.mp_earth, d.ap_AU, d.ep, d.dp_cgs)

# Run simulation (values can be NASA-filled or user-tweaked)
@app.callback(
    Output("orbit-graph", "figure"),
    Input("run-btn", "n_clicks"),
    State("Ts", "value"), State("rs_solar", "value"), State("ms_solar", "value"),
    State("mp_earth", "value"), State("dp_cgs", "value"),
    State("ap_AU", "value"), State("ep", "value"),
    State("mm_earth", "value"), State("am_hill", "value"), State("em", "value"),
    prevent_initial_call=True,  # do not block initial render
)
def run_cb(n_clicks, Ts, rs_solar, ms_solar, mp_earth, dp_cgs, ap_AU, ep, mm_earth, am_hill, em):
    if not n_clicks:
        return _initial_figure()

    d = SystemParams()
    p = SystemParams(
        Ts=_fnum(Ts, d.Ts), rs_solar=_fnum(rs_solar, d.rs_solar), ms_solar=_fnum(ms_solar, d.ms_solar),
        mp_earth=_fnum(mp_earth, d.mp_earth), dp_cgs=_fnum(dp_cgs, d.dp_cgs),
        ap_AU=_fnum(ap_AU, d.ap_AU), ep=_fnum(ep, d.ep),
        mm_earth=_fnum(mm_earth, d.mm_earth), am_hill=_fnum(am_hill, d.am_hill), em=_fnum(em, d.em),
    )
    sim = run_simulation(p)
    fig = build_animation(sim["traj"], sim["a_inner_au"], sim["a_outer_au"], open_in_browser=False)

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