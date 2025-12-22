# ------ Overview ------

# run_dash.py hosts the Dash web UI. It renders controls, 
# launches simulations either locally or via AWS Step Functions, 
# persists results in Dash Stores, and draws animations and EDA plots.

from datetime import datetime
import os, sys, uuid, json, re, io
import urllib.parse as _url
import numpy as np
import pandas as pd
import boto3
import botocore
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, ctx, no_update

from exomoon.habitable_zone import hz_bounds_au
from exomoon.constants import rsun

print(f"[ENV DEBUG] AWS_ENABLED={os.getenv('AWS_ENABLED')}", flush=True)
print(f"[ENV DEBUG] EXOMOON_BUCKET={os.getenv('EXOMOON_BUCKET')}", flush=True)
print(f"[ENV DEBUG] STATE_MACHINE_ARN={os.getenv('STATE_MACHINE_ARN')}", flush=True)
print(f"[ENV DEBUG] All set: {bool(os.getenv('AWS_ENABLED') == '1' and os.getenv('EXOMOON_BUCKET') and os.getenv('STATE_MACHINE_ARN'))}", flush=True)

AWS_ENABLED = os.getenv("AWS_ENABLED", "0") == "1"
BUCKET = os.getenv("EXOMOON_BUCKET")
STATE_MACHINE_ARN = os.getenv("STATE_MACHINE_ARN")
AWS_REGION = os.getenv("AWS_REGION", "eu-west-2")

print(f"[ENV] AWS_ENABLED={AWS_ENABLED}, BUCKET={BUCKET}, REGION={AWS_REGION}", flush=True)

if AWS_ENABLED and BUCKET and STATE_MACHINE_ARN:
    s3 = boto3.client('s3', region_name=AWS_REGION)
    sf = boto3.client('stepfunctions', region_name=AWS_REGION)
    try:
        s3.head_bucket(Bucket=BUCKET)
        print(f"[SUCCESS] S3 + credentials working", flush=True)
    except Exception as e:
        print(f"[FAIL] S3 test: {e}", flush=True)
        s3 = sf = None
else:
    s3 = sf = None

SRC_DIR = os.path.dirname(__file__)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from exomoon.params import SystemParams
from exomoon.simulation import run_simulation
from exomoon.plotting.anim import build_animation
from exomoon.exoplanet_archive import fetch_system_by_planet, estimate_density_gcc, search_planets
from exomoon.eda import pack_sim, unpack_sim, traj_to_frame, to_csv_bytes, var_info

app = Dash(__name__, suppress_callback_exceptions=True,
    title="Exomoon Orbital Integrator (Interactive)"
)

# Allow duplicate outputs/callbacks to fire on initial render safely
app.config["prevent_initial_callbacks"] = "initial_duplicate"

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
], style={"display": "flex", "flexDirection": "column", "gap": "10px"})


# A placeholder figure shown before any run. Used to keep the graph non-empty.
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


# Wraps the orbit-graph figure in a Loading indicator.
def _main_page():
    return html.Div([
        dcc.Loading(
            id="loading",
            type="default",
            children=dcc.Graph(
                id="orbit-graph",
                figure=_initial_figure(),
                style={"height": "90vh"},
                config={"scrollZoom": True, "displayModeBar": True, "responsive": True}
            )
        )
    ])


# Two-column view (controls + plot) designed to be stable in hosted environments (flex-based). 
# EDA plots render from simdata without requiring an active animation.
def _eda_page():
    return html.Div([
        html.H3("Simulation EDA"),
        html.Div([
            html.Button("Back to Simulation", id="back-btn", n_clicks=0, style={"marginRight": "12px"}),
            html.Div(id="eda-status", style={"display": "inline-block", "color": "#444"})
        ], style={"marginBottom": "12px"}),

        # Wrap controls + graph in a flex row (prevents compression)
        html.Div([
            html.Div([
                html.Label("Variables (Y-axis, multi-select)"),
                dcc.Dropdown(id="eda-vars", options=[], multi=True, placeholder="Select variables",
                             persistence=True, persisted_props=["value"]),
                html.Label("Plot type"),
                dcc.RadioItems(
                    id="eda-plot-type",
                    options=[{"label": "Line", "value": "line"}, {"label": "Scatter", "value": "scatter"}],
                    value="line", inline=True, style={"marginTop": "6px", "marginBottom": "12px"}
                ),
                html.Label("Normalize variables (divide by max)"),
                dcc.Checklist(id="eda-normalize", options=[{"label": "Normalize", "value": "norm"}], value=[]),
            ], style={"width": "340px", "flex": "0 0 340px"}),

            html.Div([
                dcc.Graph(
                    id="eda-graph",
                    style={"height": "80vh", "backgroundColor": "white"},
                    config={"scrollZoom": True, "displayModeBar": True, "responsive": True}
                )
            ], style={"flex": "1 1 auto", "minWidth": "0"})
        ], style={"display": "flex", "gap": "16px"})
    ])

app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    dcc.Store(id="kick", data=""), # control to trigger autoruns (="run") or navigation cleanup ("")    
    dcc.Store(id="simdata", data=""), # packed simulation data (arrays + timestep (dt) + simulation time (t_end), Habitable Zone)
    dcc.Store(id="job-info", data=None),
    dcc.Interval(id="status-interval", interval=5000, n_intervals=0, disabled=True),
    dcc.Download(id="download-csv"),
    html.Div(controls, style={"width": "380px", "padding": "12px", "borderRight": "1px solid #ddd"}),
    html.Div(id="status-display", style={"marginTop": "8px"}),
    html.Div(id="page-content", children=_main_page(), style={"flex": "1", "padding": "12px"})
], style={"display": "flex", "height": "100vh", "fontFamily": "Segoe UI, Arial"})


# Returns the appropriate page (main vs. EDA) based on dcc.Location pathname.
@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def route(pathname):
    if pathname == "/eda":
        return _eda_page()
    return _main_page()

# Safe numeric coercion for parameter inputs.
def _fnum(v, default):
    try:
        return default if v is None or (isinstance(v, str) and v.strip() == "") else float(v)
    except Exception:
        return default

# Parses a float from a query string or text inputs. Returns the current value if parsing fails.
def _parse_floatish(val, cur):
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


# Queries the NASA Exoplanet Archive upon from Dash UI search input (textbox)   
# and returns a ranked list of matching planet names from the dropdown.
@app.callback(
    Output("pl_picker", "options"),
    Input("pl_picker", "search_value"),
    State("pl_picker", "value"),
    prevent_initial_call=False,
)
def planet_typeahead(search_value, current_value):
    q = (search_value or "").strip()
    if len(q) < 3:
        if current_value:
            return [{"label": current_value, "value": current_value}]
        return []
    try:
        names = search_planets(q, limit=50)
        if current_value and current_value not in names:
            names = [current_value] + names
        seen = set(); ordered = []
        for n in names:
            if n not in seen:
                seen.add(n); ordered.append(n)
        return [{"label": n, "value": n} for n in ordered[:25]]
    except Exception:
        return [{"label": current_value, "value": current_value}] if current_value else []


# 1. Reads the URL query (?pl=..., numeric params, moon_dir, years, run=1) based on 
# system selected from the dropdown name options.
# 2. Fetches planet system data from the archive and fills controls.
# 3. Sets kick="run" only when run=1 is in the query AND no simdata exists yet (prevents unintended re-runs later).
# 4. Returns a tuple to populate all control values and kick.

@app.callback(
    [
        Output("fetch-status", "children"),
        Output("pl_picker", "value"),
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
    State("simdata", "data"),  # Check if we already have a sim
    prevent_initial_call=False,
)
def populate_from_url_or_nasa(url_search, n_clicks, pl_value, mm_val, ah_val, em_val, dir_val, years_val, packed):
    d = SystemParams()
    status = ""
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
                kick = "run" if not packed else ""  # only autorun if no simdata yet
            return (status, pl_value, Ts, rs, ms, mp, ap, ep, dp, mm, ah, em, moon_dir, sim_years, kick)
        except Exception:
            pass

    picked = (pl_value or "").strip()
    if picked and not n_clicks:
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
    return (status, pl_value, Ts, rs, ms, mp, ap, ep, dp, mm, ah, em, moon_dir, sim_years, kick)

# Launches a simulation either locally or via AWS Step Functions based on the Run button click or kick="run".

# Guards: Uses ctx.triggered_id to ignore navigation/cleanup changes to kick and to 
# ignore run-button remounts with n_clicks=0. 
# Only proceeds on an actual Run click or explicit autorun.

# ------- AWS Path (AWS_ENABLED = 1) ------ 
# 1. Writes params.json to S3 under inputs/job_id prefix.
# 2. Starts Step Functions execution with input/output S3 prefixes.
# 3. Computes HZ bounds and stores them in job-info.
# 4. Returns a placeholder figure, clears kick, enables the status poller, 
# and clears the URL query so navigation won’t re-trigger jobs.

# ------- Local Path (AWS_ENABLED = 0) ------
# Runs either run_simulation_for_years or run_simulation, 
# builds the animation figure, annotates with Hill radius and moon semi-major axis, 
# packs sim into simdata, and returns the figure with simdata set, poller disabled.

@app.callback(
    [Output("orbit-graph", "figure", allow_duplicate=True),
     Output("simdata", "data"),
     Output("job-info", "data"),
     Output("kick", "data", allow_duplicate=True),
     Output("status-interval", "disabled", allow_duplicate=True),
     Output("url", "search", allow_duplicate=True)],
    [Input("run-btn", "n_clicks"), Input("kick", "data")],
    State("Ts", "value"), State("rs_solar", "value"), State("ms_solar", "value"),
    State("mp_earth", "value"), State("dp_cgs", "value"),
    State("ap_AU", "value"), State("ep", "value"),
    State("mm_earth", "value"), State("am_hill", "value"), State("em", "value"),
    State("moon_dir", "value"), State("sim_years", "value"),
    prevent_initial_call=True,
)
def run_cb(n_clicks, kick, Ts, rs_solar, ms_solar, mp_earth, dp_cgs, ap_AU, ep,
           mm_earth, am_hill, em, moon_dir, sim_years):
    
    # Only proceed if the Run button was clicked, or we explicitly set kick="run"
    trigger = getattr(ctx, "triggered_id", None)
    if trigger == "kick" and kick != "run":
        return no_update, no_update, no_update, no_update, no_update, no_update
    if trigger == "run-btn" and ((n_clicks or 0) < 1):
        return no_update, no_update, no_update, no_update, no_update, no_update
    
    if not (n_clicks or (kick == "run")):
        return _initial_figure(), "", None, "", True, ""
    d = SystemParams()
    p = SystemParams(
        Ts=_fnum(Ts, d.Ts), rs_solar=_fnum(rs_solar, d.rs_solar), ms_solar=_fnum(ms_solar, d.ms_solar),
        mp_earth=_fnum(mp_earth, d.mp_earth), dp_cgs=_fnum(dp_cgs, d.dp_cgs),
        ap_AU=_fnum(ap_AU, d.ap_AU), ep=_fnum(ep, d.ep),
        mm_earth=_fnum(mm_earth, d.mm_earth), am_hill=_fnum(am_hill, d.am_hill), em=_fnum(em, d.em),
        moon_retrograde=(moon_dir == "retro"),
    )
    yrs = _fnum(sim_years, 0.0)

    if s3 and sf:
        try:
            job_id = f"job-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
            input_prefix = f"exomoon_orbital_integrator/inputs/{job_id}/"
            params_dict = {
                "Ts": p.Ts, "rs_solar": p.rs_solar, "ms_solar": p.ms_solar,
                "mp_earth": p.mp_earth, "dp_cgs": p.dp_cgs, "ap_AU": p.ap_AU, "ep": p.ep,
                "mm_earth": p.mm_earth, "am_hill": p.am_hill, "em": p.em,
                "moon_retrograde": p.moon_retrograde, "years": yrs
            }
            print(f"[DEBUG] AWS job_id={job_id}, bucket={BUCKET}")
            s3.put_object(Bucket=BUCKET, Key=f"{input_prefix}params.json", Body=json.dumps(params_dict))
            output_prefix = f"exomoon_orbital_integrator/outputs/{job_id}/"
            resp = sf.start_execution(
                stateMachineArn=STATE_MACHINE_ARN,
                name=job_id,
                input=json.dumps({
                    "inputS3Prefix": f"s3://{BUCKET}/{input_prefix}",
                    "outputS3Prefix": f"s3://{BUCKET}/{output_prefix}"
                })
            )

            hz_inner, hz_outer = hz_bounds_au(p.Ts, p.rs_solar * rsun)
            if hz_inner > hz_outer:
                hz_inner, hz_outer = hz_outer, hz_inner

            job_info = {
                "job_id": job_id,
                "execution_arn": resp["executionArn"],
                "output_prefix": output_prefix,
                "bucket": BUCKET,
                "status": "RUNNING",
                "a_inner_au": float(hz_inner),
                "a_outer_au": float(hz_outer),
                "Ts": float(p.Ts),
                "rs_solar": float(p.rs_solar),
            }

            fig = _initial_figure()
            fig.update_layout(annotations=[dict(
                text=f"🚀 AWS Job {job_id} launched!<br>Execution: {resp.get('executionArn')}",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, font=dict(size=14)
            )])
            # Clear kick, enable poller, clear URL query
            return fig, "", job_info, "", False, ""
        except Exception as e:
            print(f"[DEBUG] AWS failed: {e}")
            fig = _initial_figure()
            fig.update_layout(annotations=[dict(
                text=f"AWS Error: {str(e)[:100]}",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, font=dict(size=14)
            )])
            return fig, "", None, "", True, ""

    # Local fallback
    if yrs > 0:
        from exomoon.simulation import run_simulation_for_years
        sim = run_simulation_for_years(p, yrs)
        duration_label = f"{yrs:.3f} years"
    else:
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
    return fig, packed, None, "", True, ""

# Unpacks packed simdata string and converts the trajectory dataframe to CSV, returning a Dash download.
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


# Extracts variables from unpacked simdata and populates the EDA variables dropdown.
@app.callback(
    [Output("eda-vars", "options"), Output("eda-status", "children")],
    Input("simdata", "data"),
    prevent_initial_call=False,
)
def load_variables(packed):
    if not packed:
        return [], "No simulation data yet. Run a simulation first."
    sim = unpack_sim(packed)
    frame = traj_to_frame(sim)
    cols = frame.columns.tolist() if hasattr(frame, "columns") else list(frame.keys())
    opts = [{"label": c, "value": c} for c in cols if c != "t_years"]
    return opts, f"Loaded {len(cols)-1} variables."


# Auto-selects sensible default variables once options are ready.
@app.callback(
    Output("eda-vars", "value"),
    Input("eda-vars", "options"),
    State("simdata", "data"),
    prevent_initial_call=True,
)
def pick_default_eda_vars(options, packed):
    if not options or not packed:
        return no_update
    present = [opt["value"] for opt in options]
    defaults = [v for v in ("moon_planet_dist", "planet_star_dist", "moon_speed") if v in present]
    if not defaults:
        defaults = present[:3]
    return defaults


# Generates the time series plot for selected variables. 
@app.callback(
    Output("eda-graph", "figure"),
    [Input("eda-vars", "value"), Input("eda-plot-type", "value"), Input("eda-normalize", "value")],
    State("simdata", "data"),
    prevent_initial_call=True,
)
def eda_plot(vars_selected, ptype, norm_opts, packed):
    if not packed or not vars_selected:
        return go.Figure(layout=dict(template="plotly_white"))
    sim = unpack_sim(packed)
    frame = traj_to_frame(sim)

    t = frame["t_years"]
    t_arr = t.to_numpy() if hasattr(t, "to_numpy") else np.asarray(t)

    vars_selected = vars_selected or []
    normalize = "norm" in (norm_opts or [])  # Normalization divides by max magnitude.

    fig = go.Figure()
    for v in vars_selected:
        if v not in frame:
            continue
        y = frame[v]
        y = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)  # robust
        if normalize:
            m = float(np.max(np.abs(y))) if y.size else 1.0
            y = y / (m if m != 0 else 1.0)
        mode = "markers" if ptype == "scatter" else "lines"
        label, _unit = var_info(v)
        fig.add_trace(go.Scatter(x=t_arr, y=y, mode=mode, name=label))

    if len(vars_selected) == 1:
        label, unit = var_info(vars_selected[0])
        title = f"{label} vs Time"
    else:
        names = [var_info(v)[0] for v in vars_selected[:3]]
        more = "" if len(vars_selected) <= 3 else f" +{len(vars_selected)-3} more"
        title = f"{', '.join(names)}{more} vs Time"

    if normalize:
        ytitle = "Normalized Value"
    else:
        units = {var_info(v)[1] for v in vars_selected}
        units.discard(None)
        if len(units) == 1:
            ytitle = f"Value ({list(units)[0]})"
        elif len(units) == 0:
            ytitle = "Value"
        else:
            ytitle = "Value (mixed units)"

    xmin = float(t_arr[0]) if len(t_arr) else 0.0
    xmax = float(t_arr[-1]) if len(t_arr) else 1.0
    fig.update_xaxes(range=[xmin, xmax])
    fig.update_yaxes(autorange=True)

    # Uses plotly_white and uirevision="eda" to keep interactions and 
    # avoid background artifacts.
    fig.update_layout(
        template="plotly_white",
        uirevision="eda",
        title=title,
        xaxis_title="Time (years)",
        yaxis_title=ytitle,
        height=800,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# Sets pathname to “/” when the “Back to Simulation” button is clicked.
@app.callback(
    Output("url", "pathname"),
    Input("back-btn", "n_clicks"),
    prevent_initial_call=True
)
def go_back(n):
    if n:
        return "/"
    return no_update

# Stops when job-info is missing or already SUCCEEDED.
# Checks S3 for output files (traj.csv or trak.csv). If found, marks SUCCEEDED.
# Otherwise calls StepFunctions describe_execution; disables poller when terminal status reached.
@app.callback(
    [Output("job-info", "data", allow_duplicate=True),
     Output("status-interval", "disabled", allow_duplicate=True)],
    Input("status-interval", "n_intervals"),
    State("job-info", "data"),
    prevent_initial_call=True,
)
def poll_job_status(n_intervals, job_info):
    if isinstance(job_info, list):
        job_info = job_info[0] if job_info else None

    if not job_info:
        return job_info, True
    if job_info.get("status") == "SUCCEEDED":
        return job_info, True

    try:
        prefix = job_info["output_prefix"]; bucket = job_info["bucket"]
        for name in ("traj.csv", "trak.csv"):
            key = f"{prefix}{name}"
            s3.head_object(Bucket=bucket, Key=key)
            job_info["status"] = "SUCCEEDED"
            print(f"[POLL] S3 ready: {key}", flush=True)
            return job_info, True
    except botocore.exceptions.ClientError as ce:
        code = ce.response.get("Error", {}).get("Code")
        if code in ("404", "NoSuchKey", "NotFound"):
            pass
        else:
            print(f"[POLL] S3 head_object error: {code}: {ce}", flush=True)
    except Exception as e:
        print(f"[POLL] S3 check error: {e}", flush=True)

    try:
        resp = sf.describe_execution(executionArn=job_info["execution_arn"])
        status = resp.get("status", job_info.get("status", "RUNNING"))
        job_info["status"] = status
        return job_info, (status in ("SUCCEEDED", "FAILED", "TIMED_OUT"))
    except Exception as e:
        print(f"[POLL] StepFunctions describe failed: {e}", flush=True)
        return job_info, False

# When job succeeds, reads traj.csv and summary.json from S3.
# Computes/falls back HZ bounds if missing.
# Rebuilds the animation figure and packs simdata (Note: Currently, does not retrieve animations.html from backend S3). 
# Enables main page to render the animation without re-running locally.
@app.callback(
    [Output("orbit-graph", "figure", allow_duplicate=True),
     Output("simdata", "data", allow_duplicate=True)],
    Input("job-info", "data"),
    State("job-info", "data"),
    prevent_initial_call=True,
)
def load_s3_results(job_info_trigger, job_info):
    if isinstance(job_info, list):
        job_info = job_info[0] if job_info else None

    print(f"[LOAD DEBUG] Triggered: status={job_info.get('status') if job_info else None}")
    if not job_info or job_info.get("status") != "SUCCEEDED":
        return no_update, no_update

    try:
        key_csv = f"{job_info['output_prefix']}traj.csv"
        csv_obj = s3.get_object(Bucket=job_info["bucket"], Key=key_csv)
        df = pd.read_csv(io.StringIO(csv_obj["Body"].read().decode("utf-8")))
        cols = list(df.columns)

        a_inner_au = job_info.get("a_inner_au")
        a_outer_au = job_info.get("a_outer_au")
        summary = None
        try:
            key_sum = f"{job_info['output_prefix']}summary.json"
            sum_obj = s3.get_object(Bucket=job_info["bucket"], Key=key_sum)
            summary = json.loads(sum_obj["Body"].read().decode("utf-8"))
            a_inner_au = summary.get("a_inner_au") or summary.get("hz_inner_au") or a_inner_au
            a_outer_au = summary.get("a_outer_au") or summary.get("hz_outer_au") or a_outer_au
        except Exception:
            summary = None

        if a_inner_au is None or a_outer_au is None:
            Ts = job_info.get("Ts"); rs_solar = job_info.get("rs_solar")
            if Ts is not None and rs_solar is not None:
                i, o = hz_bounds_au(float(Ts), float(rs_solar) * rsun)
                a_inner_au = a_inner_au or i
                a_outer_au = a_outer_au or o

        if a_inner_au is None or a_outer_au is None:
            a_inner_au, a_outer_au = 0.75, 1.75
        if a_inner_au > a_outer_au:
            a_inner_au, a_outer_au = a_outer_au, a_inner_au

        xyz_mp = df[["planet_x", "planet_y", "planet_z"]].values if all(c in cols for c in ["planet_x", "planet_y", "planet_z"]) else np.zeros((1000, 3))
        xyz_ms = df[["star_x", "star_y", "star_z"]].values   if all(c in cols for c in ["star_x", "star_y", "star_z"]) else np.zeros((1000, 3))
        xyz_mm = df[["moon_x", "moon_y", "moon_z"]].values   if all(c in cols for c in ["moon_x", "moon_y", "moon_z"]) else np.zeros((1000, 3))

        vel_mp = df[["planet_vx", "planet_vy", "planet_vz"]].values if all(c in cols for c in ["planet_vx", "planet_vy", "planet_vz"]) else None
        vel_ms = df[["star_vx", "star_vy", "star_vz"]].values   if all(c in cols for c in ["star_vx", "star_vy", "star_vz"]) else None
        vel_mm = df[["moon_vx", "moon_vy", "moon_vz"]].values   if all(c in cols for c in ["moon_vx", "moon_vy", "moon_vz"]) else None

        dt = float(np.diff(df["t_years"][:2])[0]) if "t_years" in df and len(df) >= 2 else 0.001
        t_end = float(df["t_years"].max() if "t_years" in df else 1.0)

        sim = {
            "traj": {
                "xyzarr_mp": xyz_mp,
                "xyzarr_ms": xyz_ms,
                "xyzarr_mm": xyz_mm,
                "velarr_mp": vel_mp,
                "velarr_ms": vel_ms,
                "velarr_mm": vel_mm,
            },
            "a_inner_au": float(a_inner_au),
            "a_outer_au": float(a_outer_au),
            "dt": dt,
            "t_end": t_end,
            "state": {"rhill_AU": (summary or {}).get("rhill_AU")},
        }

        fig = build_animation(sim["traj"], sim["a_inner_au"], sim["a_outer_au"],
                              open_in_browser=False, dt=sim["dt"], t_end=sim["t_end"])
        packed = pack_sim(sim)
        return fig, packed

    except Exception as e:
        print(f"[RESULTS] Failed to load {job_info.get('job_id')}: {e}", flush=True)
        return _initial_figure(), ""


# Renders a small status banner with a link to the S3 output prefix when SUCCEEDED.
@app.callback(
    Output("status-display", "children"),
    Input("job-info", "data"),
    prevent_initial_call=True,
)
def update_status_display(job_info):
    if isinstance(job_info, list):
        job_info = job_info[0] if job_info else None
    if not job_info:
        return ""
    status = job_info.get("status", "UNKNOWN")
    job_id = job_info.get("job_id", "unknown")
    if status == "SUCCEEDED":
        return html.Div([
            html.Span(f"✅ Job {job_id} completed! ", style={"color": "green"}),
            html.A("View S3 outputs", href=f"https://{BUCKET}.s3.amazonaws.com/{job_info['output_prefix']}", target="_blank")
        ])
    elif status in ["FAILED", "TIMED_OUT"]:
        return html.Span(f"❌ Job {job_id} {status}", style={"color": "red"})
    else:
        return html.Span(f"⏳ Job {job_id}: {status}", style={"color": "orange"})

# Clears the URL query string on any navigation so run=1 doesn’t persist.
@app.callback(
    Output("url", "search", allow_duplicate=True),
    Input("url", "pathname"),
    prevent_initial_call=False,
)
def clear_query_on_route(pathname):
    # Remove ?run=1 or any other query when changing pages
    return ""

# Clears the kick store on navigation to the EDA page to avoid unintended re-runs
# due to stale kicks on route changes.
@app.callback(
    Output("kick", "data", allow_duplicate=True),
    Input("url", "pathname"),
    prevent_initial_call=False,
)
def clear_kick_on_route(pathname):
    if pathname == "/eda":
        return ""
    return no_update

# When navigating back to “/”, rebuilds the orbit-graph from simdata to restore main figure. 
# Uses plotly_white and uirevision="anim" to stabilize visuals and preserve interactions (zoom state etc.).
@app.callback(
    Output("orbit-graph", "figure", allow_duplicate=True),
    [Input("url", "pathname"), Input("simdata", "data")],
    prevent_initial_call=False,   # restore immediately on return to "/"
)
def restore_main_figure(pathname, packed):
    if pathname != "/":
        return no_update
    if not packed:
        return no_update
    sim = unpack_sim(packed)
    a_in = sim.get("a_inner_au") or 0.75
    a_out = sim.get("a_outer_au") or 1.75
    if a_in > a_out:
        a_in, a_out = a_out, a_in
    fig = build_animation(sim["traj"], a_in, a_out, open_in_browser=False, dt=sim["dt"], t_end=sim["t_end"])
    fig.update_layout(template="plotly_white", uirevision="anim")
    return fig

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8050"))
    debug = os.getenv("DEBUG", "1") == "1"
    app.run(host=host, port=port, debug=debug)

#if __name__ == "__main__":
#    app.run(debug=True)
    
    
    #print(f"Running Dash server on port: {8080}")
    #app.run(host="0.0.0.0", port=8080, debug=True)
