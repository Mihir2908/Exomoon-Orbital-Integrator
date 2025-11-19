import re
import urllib.parse as _url
import os, sys
import numpy as np
from dash import Dash, dcc, html, Input, Output, State, ctx, no_update
import plotly.graph_objects as go

# Ensure src/ is importable regardless of current working dir
SRC_DIR = os.path.dirname(__file__)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from exomoon.params import SystemParams
from exomoon.simulation import run_simulation
from exomoon.plotting.anim import build_animation
from exomoon.exoplanet_archive import fetch_system_by_planet, estimate_density_gcc, search_planets
from exomoon.eda import pack_sim, unpack_sim, traj_to_frame, to_csv_bytes, var_info

app = Dash(__name__, suppress_callback_exceptions=True,  # allow callbacks for components added later (/eda)
    title="Exomoon Orbital Integrator (Interactive)"
)

_defaults = SystemParams()

controls = html.Div([
    html.H4("NASA Exoplanet Archive"),
    html.Div([
        dcc.Dropdown(
            id="pl_picker",
            options=[],
            placeholder="Type 3+ characters to search…",
            clearable=True,
            searchable=True,
            style={"width": "100%"}
        ),
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
    html.Button("Export CSV", id="export-btn", n_clicks=0, style={"width": "100%", "marginTop": "6px"}),
    dcc.Link("Open EDA Plots →", href="/eda", style={"marginTop": "8px", "textDecoration": "none", "color": "#0074D9"}),
], style={"display": "flex", "flexDirection": "column", "gap": "10px"}),


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

# NEW: main vs EDA page containers
def _main_page():
    return html.Div([
        dcc.Loading(
            id="loading",
            type="default",
            children=dcc.Graph(id="orbit-graph", figure=_initial_figure(), style={"height": "90vh"})
        )
    ])

def _eda_page():
    return html.Div([
        html.H3("Simulation EDA"),
        html.Div([
            html.Button("Back to Simulation", id="back-btn", n_clicks=0, style={"marginRight": "12px"}),
            html.Div(id="eda-status", style={"display": "inline-block", "color": "#444"})
        ], style={"marginBottom": "12px"}),
        html.Div([
            html.Label("Variables (Y-axis, multi-select)"),
            dcc.Dropdown(id="eda-vars", options=[], multi=True, placeholder="Select variables"),
            html.Label("Plot type"),
            dcc.RadioItems(
                id="eda-plot-type",
                options=[{"label": "Line", "value": "line"}, {"label": "Scatter", "value": "scatter"}],
                value="line", inline=True, style={"marginTop": "6px", "marginBottom": "12px"}
            ),
            html.Label("Normalize variables (divide by max)"),
            dcc.Checklist(id="eda-normalize", options=[{"label": "Normalize", "value": "norm"}], value=[]),
        ], style={"width": "340px", "float": "left", "marginRight": "24px"}),
        html.Div([
            dcc.Graph(id="eda-graph", style={"height": "80vh"})
        ], style={"overflow": "hidden"}),
        # Hidden orbit graph so run_cb Output is always valid
        dcc.Graph(id="orbit-graph", figure=_initial_figure(), style={"display": "none"})
    ])


app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    dcc.Store(id="kick", data=""),
    dcc.Store(id="simdata", data=""),
    dcc.Download(id="download-csv"),
    html.Div(controls, style={"width": "380px", "padding": "12px", "borderRight": "1px solid #ddd"}),
    html.Div(id="page-content", children=_main_page(), style={"flex": "1", "padding": "12px"})
], style={"display": "flex", "height": "100vh", "fontFamily": "Segoe UI, Arial"})

# NEW: router
@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def route(pathname):
    if pathname == "/eda":
        return _eda_page()
    return _main_page()

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

# NEW: populate suggestions when typing in the textbox
@app.callback(
    Output("pl_picker", "options"),
    Input("pl_picker", "search_value"),
    State("pl_picker", "value"),
    prevent_initial_call=False,
)
def planet_typeahead(search_value, current_value):
    q = (search_value or "").strip()
    # If user cleared the search box or fewer than 3 chars: keep current selection visible
    if len(q) < 3:
        if current_value:
            return [{"label": current_value, "value": current_value}]
        return []
    try:
        names = search_planets(q, limit=50)
        # Ensure current selection stays in list
        if current_value and current_value not in names:
            names = [current_value] + names
        # Deduplicate while preserving order
        seen = set()
        ordered = []
        for n in names:
            if n not in seen:
                seen.add(n)
                ordered.append(n)
        return [{"label": n, "value": n} for n in ordered[:25]]
    except Exception:
        return [{"label": current_value, "value": current_value}] if current_value else []


# Fetch from NASA and fill sliders
@app.callback(
    [
        Output("fetch-status", "children"),
        Output("pl_picker", "value"),            # NEW: keep dropdown selection in sync (URL, fetch)
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
        Output("moon_dir", "value"),
        Output("sim_years", "value"),
        Output("kick", "data"),
    ],
    [Input("url", "search"), Input("fetch-btn", "n_clicks"), Input("pl_picker", "value")],
    State("mm_earth", "value"), State("am_hill", "value"), State("em", "value"),
    State("moon_dir", "value"),
    State("sim_years", "value"),
    prevent_initial_call=False,
)
def populate_from_url_or_nasa(url_search, n_clicks, pl_value, mm_val, ah_val, em_val, dir_val, years_val):
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
    kick = ""
    moon_dir = dir_val or "pro"

    # URL path
    if url_search:
        try:
            q = {k: v[0] for k, v in _url.parse_qs(url_search.lstrip("?")).items()}
            pl = q.get("pl")
            if pl and pl.strip():
                try:
                    rec = fetch_system_by_planet(pl)
                    if rec:
                        status = f"Loaded: {rec.get('pl_name')} (host: {rec.get('hostname')})"
                        pl_value = rec.get("pl_name") or pl
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
                kick = "run"
            return (status, pl_value, Ts, rs, ms, mp, ap, ep, dp, mm, ah, em, moon_dir, sim_years, kick)
        except Exception:
            pass

    # Selection fetch (dropdown value chosen)
    picked = (pl_value or "").strip()
    if picked and not n_clicks:  # avoid double-fetch if button also clicked
        try:
            rec = fetch_system_by_planet(picked)
            if rec:
                status = f"Loaded: {rec.get('pl_name')} (host: {rec.get('hostname')})"
                pl_value = rec.get("pl_name") or picked
                Ts = rec.get("Ts") or Ts
                rs = rec.get("rs_solar") or rs
                ms = rec.get("ms_solar") or ms
                mp = rec.get("mp_earth") or mp
                ap = rec.get("ap_AU") or ap
                ep = rec.get("ep") or ep
                est_rho = estimate_density_gcc(rec.get("mp_earth"), rec.get("pl_rade"))
                dp = est_rho or dp
                return (status, pl_value, Ts, rs, ms, mp, ap, ep, dp, mm, ah, em, moon_dir, sim_years, kick)
            else:
                status = f"No results for '{picked}'."
                return (status, pl_value, Ts, rs, ms, mp, ap, ep, dp, mm, ah, em, moon_dir, sim_years, kick)
        except Exception as e:
            status = f"Error fetching '{picked}': {e}"
            return (status, pl_value, Ts, rs, ms, mp, ap, ep, dp, mm, ah, em, moon_dir, sim_years, kick)

    # B) Button click flow (don't change moon sliders)
    if (n_clicks or 0) > 0:
        if not picked:
            return ("Select a planet first (type 3+ chars).", pl_value, Ts, rs, ms, mp, ap, ep, dp, mm, ah, em, moon_dir, sim_years, kick)
        try:
            rec = fetch_system_by_planet(picked)
            if not rec:
                return (f"No results for '{picked}'.", pl_value, Ts, rs, ms, mp, ap, ep, dp, mm, ah, em, moon_dir, sim_years, kick)
            status = f"Loaded: {rec.get('pl_name')} (host: {rec.get('hostname')})"
            pl_value = rec.get("pl_name") or picked
            Ts = rec.get("Ts") or Ts
            rs = rec.get("rs_solar") or rs
            ms = rec.get("ms_solar") or ms
            mp = rec.get("mp_earth") or mp
            ap = rec.get("ap_AU") or ap
            ep = rec.get("ep") or ep
            est_rho = estimate_density_gcc(rec.get("mp_earth"), rec.get("pl_rade"))
            dp = est_rho or dp
        except Exception as e:
            status = f"Error fetching '{picked}': {e}"
    #Default/first load fallback (must return all 14 values)
    return (status, pl_value, Ts, rs, ms, mp, ap, ep, dp, mm, ah, em, moon_dir, sim_years, kick)

# UPDATED: Simulation callback now also packs sim for export/EDA
@app.callback(
    [Output("orbit-graph", "figure"), Output("simdata", "data")],
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
    if not (n_clicks or (kick == "run")):
        return _initial_figure(), ""
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
        from exomoon.simulation import run_simulation
        sim = run_simulation(p)
        duration_label = "1 planet orbit"

    fig = build_animation(sim["traj"], sim["a_inner_au"], sim["a_outer_au"],
                          open_in_browser=False, dt=sim["dt"], t_end=sim["t_end"])
    st = sim.get("state", {})
    rhill = st.get("rhill_AU")
    if isinstance(rhill, (int, float)):
        dir_txt = "Retrograde" if p.moon_retrograde else "Prograde"
        fig.add_annotation(
            text=f"Hill radius: {rhill:.4f} AU | a_moon≈{p.am_hill*rhill:.4f} AU | {dir_txt} | {duration_label}",
            xref="paper", yref="paper", x=0.01, y=1.06,
            showarrow=False, align="left", font=dict(size=12)
        )
    packed = pack_sim(sim)
    return fig, packed

# NEW: CSV export button
@app.callback(
    Output("download-csv", "data"),
    Input("export-btn", "n_clicks"),
    State("simdata", "data"),
    prevent_initial_call=True,
)
def export_csv(n_clicks, packed):
    if not packed:
        return no_update
    sim = unpack_sim(packed)
    frame = traj_to_frame(sim)
    csv_bytes = to_csv_bytes(frame)
    return dcc.send_bytes(csv_bytes, "exomoon_simulation.csv")

# NEW: Populate EDA var list when we have data
@app.callback(
    [Output("eda-vars", "options"), Output("eda-status", "children")],
    [Input("simdata", "data"), Input("url", "pathname")],  # also trigger on navigation to /eda
    prevent_initial_call=False,
)
def load_variables(packed, pathname):
    if not packed:
        return [], "No simulation data yet. Run a simulation first."
    sim = unpack_sim(packed)
    frame = traj_to_frame(sim)
    cols = frame.columns.tolist() if hasattr(frame, "columns") else list(frame.keys())
    opts = [{"label": c, "value": c} for c in cols if c != "t_years"]
    return opts, f"Loaded {len(cols)-1} variables."

# NEW: EDA plot builder
@app.callback(
    Output("eda-graph", "figure"),
    [Input("eda-vars", "value"), Input("eda-plot-type", "value"), Input("eda-normalize", "value")],
    State("simdata", "data"),
    prevent_initial_call=True,
)

def eda_plot(vars_selected, ptype, norm_opts, packed):
    import plotly.graph_objects as go
    if not packed or not vars_selected:
        return go.Figure()
    sim = unpack_sim(packed)
    frame = traj_to_frame(sim)
    t = frame["t_years"]
    # Use positional array to avoid pandas label indexing for -1
    t_arr = t.to_numpy() if hasattr(t, "to_numpy") else np.asarray(t)

    vars_selected = vars_selected or []
    normalize = "norm" in (norm_opts or [])

    fig = go.Figure()
    for v in vars_selected:
        if v not in frame:
            continue
        y = frame[v]
        if normalize:
            m = float(np.max(np.abs(y))) if len(y) else 1.0
            y = y / (m if m != 0 else 1.0)
        mode = "markers" if ptype == "scatter" else "lines"
        label, _unit = var_info(v)
        fig.add_trace(go.Scatter(x=t_arr, y=y, mode=mode, name=label))

    # Dynamic title
    if len(vars_selected) == 1:
        label, unit = var_info(vars_selected[0])
        title = f"{label} vs Time"
    else:
        names = [var_info(v)[0] for v in vars_selected[:3]]
        more = "" if len(vars_selected) <= 3 else f" +{len(vars_selected)-3} more"
        title = f"{', '.join(names)}{more} vs Time"

    # Y-axis label with units (if all same and not normalized)
    if normalize:
        ytitle = "Normalized Value"
    else:
        units = {var_info(v)[1] for v in vars_selected if v in frame}
        units.discard(None)
        if len(units) == 1:
            ytitle = f"Value ({list(units)[0]})"
        elif len(units) == 0:
            ytitle = "Value"
        else:
            ytitle = "Value (mixed units)"

    # Axes ranges: always show full time
    xmin = float(t_arr[0]) if len(t_arr) else 0.0
    xmax = float(t_arr[-1]) if len(t_arr) else 1.0
    fig.update_xaxes(range=[xmin, xmax])

    if len(vars_selected) == 1 and vars_selected[0] in frame:
        y0 = frame[vars_selected[0]]
        ymin, ymax = float(np.min(y0)), float(np.max(y0))
        if ymin == ymax:
            pad = 1.0 if ymax == 0.0 else 0.05 * abs(ymax)
            ymin -= pad; ymax += pad
        else:
            span = ymax - ymin
            pad = 0.05 * span
            ymin -= pad; ymax += pad
        fig.update_yaxes(range=[ymin, ymax])
    else:
        fig.update_yaxes(autorange=True)

    fig.update_layout(
        title=title,
        xaxis_title="Time (years)",
        yaxis_title=ytitle,
        height=800,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# NEW: Back button to navigate to main page
@app.callback(
    Output("url", "pathname"),
    Input("back-btn", "n_clicks"),
    prevent_initial_call=True
)
def go_back(n):
    if n:
        return "/"
    return no_update


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8050"))
    debug = os.getenv("DEBUG", "1") == "1"
    app.run(host=host, port=port, debug=debug)

#if __name__ == "__main__":
#    app.run(debug=True)
    
    
    #print(f"Running Dash server on port: {8080}")
    #app.run(host="0.0.0.0", port=8080, debug=True)
