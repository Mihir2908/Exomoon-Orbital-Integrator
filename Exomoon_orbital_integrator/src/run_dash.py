import re
import urllib.parse as _url
import os, sys
from dash import Dash, dcc, html, Input, Output, State, ctx
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

    html.Label("Moon orbit direction"),
    dcc.RadioItems(
        id="moon_dir",
        options=[
            {"label": "Prograde", "value": "pro"},
            {"label": "Retrograde", "value": "retro"},
        ],
        value="pro",
        inline=True,
        style={"marginBottom": "6px"}
    ),
    html.Label("Simulation duration (years, 0 = 1 planet orbit)"),
    dcc.Input(id="sim_years", type="number", value=0, min=0, step="any", style={"width": "100%"}),

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
    dcc.Location(id="url", refresh=False),
    dcc.Store(id="kick", data=""),
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

def _parse_floatish(val, cur):
    # Accept numbers or strings with units (e.g., "2.54 M_earth", "0.4 R_hill")
    if val is None:
        return cur
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', val.strip())
        if m:
            try:
                return float(m.group(0))
            except Exception:
                return cur
    return cur

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
        Output("mm_earth", "value"),
        Output("am_hill", "value"),
        Output("em", "value"),
        Output("moon_dir", "value"),  # ensure radio reflects URL or stays as-is
        Output("sim_years", "value"),            # (for simulation runtime)
        Output("kick", "data"),                   # (for autorun)
    
    ],
    [Input("url", "search"), Input("fetch-btn", "n_clicks")],
    State("pl_name", "value"),
    State("mm_earth", "value"), State("am_hill", "value"), State("em", "value"),
    State("moon_dir", "value"),
    State("sim_years", "value"),
    prevent_initial_call=False,
)
def populate_from_url_or_nasa(url_search, n_clicks, pl_name, mm_val, ah_val, em_val, dir_val, years_val):
    d = SystemParams()
    status = ""

    # Start with current (or defaults on first load)
    def _fv(v, default):
        try:
            return default if v is None or (isinstance(v, str) and v.strip() == "") else float(v)
        except Exception:
            return default

    Ts, rs, ms, mp, ap, ep, dp = d.Ts, d.rs_solar, d.ms_solar, d.mp_earth, d.ap_AU, d.ep, d.dp_cgs
    mm, ah, em = _fv(mm_val, d.mm_earth), _fv(ah_val, d.am_hill), _fv(em_val, d.em)
    sim_years = _fv(years_val, 0.0)
    kick = ""  # empty means no autorun
    def _fo(val, cur):
        try:
            return cur if val is None else float(val)
        except Exception:
            return cur

    moon_dir = dir_val or "pro"
    # A) URL path (on load or when URL changes)
    if url_search:
        try:
            q = {k: v[0] for k, v in _url.parse_qs(url_search.lstrip("?")).items()}
            # Fetch NASA if planet given
            pl = q.get("pl")
            if pl and pl.strip():
                try:
                    rec = fetch_system_by_planet(pl)
                    if rec:
                        status = f"Loaded: {rec.get('pl_name')} (host: {rec.get('hostname')})"
                        Ts = rec.get("Ts") or Ts
                        rs = rec.get("rs_solar") or rs
                        ms = rec.get("ms_solar") or ms
                        mp = rec.get("mp_earth") or mp
                        ap = rec.get("ap_AU") or ap
                        ep = rec.get("ep") or ep
                        est_rho = estimate_density_gcc(rec.get("mp_earth"), rec.get("pl_rade"))
                        dp = est_rho or dp
                    else:
                        status = f"No results for '{pl}'."
                except Exception as e:
                    status = f"Error fetching '{pl}': {e}"
            # Numeric overrides
            Ts  = _parse_floatish(q.get("Ts"), Ts);  rs  = _parse_floatish(q.get("rs_solar"), rs)
            ms  = _parse_floatish(q.get("ms_solar"), ms); mp  = _parse_floatish(q.get("mp_earth"), mp)
            ap  = _parse_floatish(q.get("ap_AU"), ap);   ep  = _parse_floatish(q.get("ep"), ep)
            dp  = _parse_floatish(q.get("dp_cgs"), dp)
            mm  = _parse_floatish(q.get("mm_earth"), mm); ah  = _parse_floatish(q.get("am_hill"), ah)
            em  = _parse_floatish(q.get("em"), em)
            sim_years = _parse_floatish(q.get("years") or q.get("t_years") or q.get("duration"), sim_years)

            md_raw = q.get("moon_dir") or q.get("moon_retrograde")
            if md_raw:
                md_raw = md_raw.lower()
                moon_dir = "retro" if md_raw in ("retro","retrograde","r","true","1") else "pro"

            if q.get("run") in ("1","true","yes"):
                kick = "run"  # signal autorun

            return (status, Ts, rs, ms, mp, ap, ep, dp, mm, ah, em, moon_dir, sim_years, kick)
        except Exception:
            pass  # fall through

    # B) Button click flow (don't change moon sliders)
    if (n_clicks or 0) > 0:
        name = (pl_name or "").strip()
        if not name:
            return ("Enter a planet name (e.g., HD 209458 b).", Ts, rs, ms, mp, ap, ep, dp, mm, ah, em, moon_dir, sim_years, kick)
        try:
            rec = fetch_system_by_planet(name)
            if not rec:
                return (f"No results for '{name}'.", Ts, rs, ms, mp, ap, ep, dp, mm, ah, em, moon_dir, sim_years, kick)
            Ts = rec.get("Ts") or Ts
            rs = rec.get("rs_solar") or rs
            ms = rec.get("ms_solar") or ms
            mp = rec.get("mp_earth") or mp
            ap = rec.get("ap_AU") or ap
            ep = rec.get("ep") or ep
            est_rho = estimate_density_gcc(rec.get("mp_earth"), rec.get("pl_rade"))
            dp = est_rho or dp
            status = f"Loaded: {rec.get('pl_name')} (host: {rec.get('hostname')})"
        except Exception as e:
            status = f"Error fetching '{name}': {e}"
    #Default/first load fallback (must return all 14 values)
    return (status, Ts, rs, ms, mp, ap, ep, dp, mm, ah, em, moon_dir, sim_years, kick)

# Simulation callback: add kick + sim_years
@app.callback(
    Output("orbit-graph", "figure"),
    [Input("run-btn", "n_clicks"), Input("kick", "data")],   # kick triggers autorun
    State("Ts", "value"), State("rs_solar", "value"), State("ms_solar", "value"),
    State("mp_earth", "value"), State("dp_cgs", "value"),
    State("ap_AU", "value"), State("ep", "value"),
    State("mm_earth", "value"), State("am_hill", "value"), State("em", "value"),
    State("moon_dir", "value"), State("sim_years", "value"),
    prevent_initial_call=True,
)
def run_cb(n_clicks, kick, Ts, rs_solar, ms_solar, mp_earth, dp_cgs, ap_AU, ep,
           mm_earth, am_hill, em, moon_dir, sim_years):
    # Only run if button clicked or autorun kick set
    if not (n_clicks or (kick == "run")):
        return _initial_figure()

    d = SystemParams()
    p = SystemParams(
        Ts=_fnum(Ts, d.Ts), rs_solar=_fnum(rs_solar, d.rs_solar), ms_solar=_fnum(ms_solar, d.ms_solar),
        mp_earth=_fnum(mp_earth, d.mp_earth), dp_cgs=_fnum(dp_cgs, d.dp_cgs),
        ap_AU=_fnum(ap_AU, d.ap_AU), ep=_fnum(ep, d.ep),
        mm_earth=_fnum(mm_earth, d.mm_earth), am_hill=_fnum(am_hill, d.am_hill), em=_fnum(em, d.em),
        moon_retrograde=(moon_dir == "retro"),
    )

    yrs = _fnum(sim_years, 0.0)
    if yrs > 0:
        from exomoon.simulation import run_simulation_for_years
        sim = run_simulation_for_years(p, yrs)
        duration_label = f"{yrs:.3f} years"
    else:
        sim = run_simulation(p)
        duration_label = "1 planet orbit"

    fig = build_animation(sim["traj"], sim["a_inner_au"], sim["a_outer_au"], open_in_browser=False, dt=sim["dt"], t_end=sim["t_end"])
    st = sim.get("state", {})
    rhill = st.get("rhill_AU")
    if isinstance(rhill, (int, float)):
        dir_txt = "Retrograde" if p.moon_retrograde else "Prograde"
        fig.add_annotation(
            text=f"Hill radius: {rhill:.4f} AU | a_moon≈{p.am_hill*rhill:.4f} AU | {dir_txt} | {duration_label}",
            xref="paper", yref="paper", x=0.01, y=1.06,
            showarrow=False, align="left", font=dict(size=12)
        )
    return fig

if __name__ == "__main__":
    app.run(debug=True)