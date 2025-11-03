from dash import Dash, dcc, html, Input, Output, State, callback
import plotly.graph_objects as go
import os, sys, importlib

# Ensure src/ is importable regardless of current working dir
SRC_DIR = os.path.dirname(__file__)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
    
#import src.exomoon.params as params
#import src.exomoon.simulation as simulation
#import src.exomoon.plotting.anim as anim
from exomoon import params
from exomoon import simulation
from exomoon import plotting    

from exomoon.params import SystemParams
from exomoon.simulation import run_simulation
from exomoon.plotting.anim import build_animation

app = Dash(__name__)
app.title = "Exomoon Orbital Integrator (Interactive)"

controls = html.Div([
    html.H4("Inputs"),
    html.Label("Stellar temperature Ts (K)"),
    dcc.Input(id="Ts", type="number", value=3784, min=2500, max=8000, step=10, style={"width": "100%"}),

    html.Label("Star radius (R☉)"),
    dcc.Slider(id="rs_solar", min=0.1, max=2.0, step=0.01, value=0.51, tooltip={"placement": "bottom"}),

    html.Label("Star mass (M☉)"),
    dcc.Slider(id="ms_solar", min=0.1, max=2.0, step=0.01, value=0.54, tooltip={"placement": "bottom"}),

    html.Label("Planet mass (M⊕)"),
    dcc.Slider(id="mp_earth", min=0.1, max=10.0, step=0.01, value=2.54, tooltip={"placement": "bottom"}),

    html.Label("Planet density (g/cc)"),
    dcc.Slider(id="dp_cgs", min=0.5, max=15.0, step=0.1, value=5.5, tooltip={"placement": "bottom"}),

    html.Label("Planet semi-major axis ap (AU)"),
    dcc.Slider(id="ap_AU", min=0.05, max=2.0, step=0.001, value=0.3006, tooltip={"placement": "bottom"}),

    html.Label("Planet eccentricity ep"),
    dcc.Slider(id="ep", min=0.0, max=0.8, step=0.01, value=0.0, tooltip={"placement": "bottom"}),

    html.Label("Moon mass (M⊕)"),
    dcc.Slider(id="mm_earth", min=0.01, max=2.0, step=0.01, value=1.0, tooltip={"placement": "bottom"}),

    html.Label("Moon a as fraction of Hill radius (am_hill)"),
    dcc.Slider(id="am_hill", min=0.1, max=0.9, step=0.01, value=0.45, tooltip={"placement": "bottom"}),

    html.Label("Moon eccentricity em"),
    dcc.Slider(id="em", min=0.0, max=0.8, step=0.01, value=0.0, tooltip={"placement": "bottom"}),

    html.Button("Run Simulation", id="run-btn", n_clicks=0, style={"width": "100%", "marginTop": "10px"}),
], style={"display": "flex", "flexDirection": "column", "gap": "10px"})

app.layout = html.Div([
    html.Div(controls, style={"width": "320px", "padding": "12px", "borderRight": "1px solid #ddd"}),
    html.Div([
        dcc.Loading(
            id="loading",
            type="default",
            children=dcc.Graph(id="orbit-graph", figure=go.Figure(), style={"height": "90vh"})
        )
    ], style={"flex": "1", "padding": "12px"})
], style={"display": "flex", "height": "100vh", "fontFamily": "Segoe UI, Arial"})

@callback(
    Output("orbit-graph", "figure"),
    Input("run-btn", "n_clicks"),
    State("Ts", "value"),
    State("rs_solar", "value"),
    State("ms_solar", "value"),
    State("mp_earth", "value"),
    State("dp_cgs", "value"),
    State("ap_AU", "value"),
    State("ep", "value"),
    State("mm_earth", "value"),
    State("am_hill", "value"),
    State("em", "value"),
    prevent_initial_call=False,  # also render defaults on first load
)
def run(n_clicks, Ts, rs_solar, ms_solar, mp_earth, dp_cgs, ap_AU, ep, mm_earth, am_hill, em):
    p = SystemParams(
        Ts=float(Ts),
        rs_solar=float(rs_solar),
        ms_solar=float(ms_solar),
        mp_earth=float(mp_earth),
        dp_cgs=float(dp_cgs),
        ap_AU=float(ap_AU),
        ep=float(ep),
        mm_earth=float(mm_earth),
        am_hill=float(am_hill),
        em=float(em),
    )
    sim = run_simulation(p)
    fig = build_animation(
        traj=sim["traj"],
        a_inner_au=sim["a_inner_au"],
        a_outer_au=sim["a_outer_au"],
        open_in_browser=False  # Dash renders in-page
    )
    return fig

if __name__ == "__main__":
    # pip install dash
    app.run(debug=True)