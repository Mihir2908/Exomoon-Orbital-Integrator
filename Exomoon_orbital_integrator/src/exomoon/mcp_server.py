from pathlib import Path
from time import perf_counter
from typing import Dict, Any
from urllib.parse import urlencode
import sys, traceback, re, json

from fastmcp import FastMCP

from exomoon.params import SystemParams
from exomoon.simulation import run_simulation, run_simulation_for_years
from exomoon.plotting.anim import build_animation
from exomoon.exoplanet_archive import fetch_system_by_planet, search_planets
from exomoon.moon_stability import assess_moon_stability as _assess_moon_stability  # import the function symbol
from exomoon.moon_stability import analyze_moon_escape as _analyze_moon_escape

mcp = FastMCP("exomoon")

def _params_from_dict(d: Dict[str, Any]) -> SystemParams:
    base = SystemParams()
    def f(k, cur):
        return cur if d.get(k) is None or d.get(k) == "" else float(d[k])
    def b(key):
        v = d.get(key)
        if v is None: return False
        if isinstance(v, bool): return v
        if isinstance(v, (int,float)): return bool(v)
        v = str(v).lower().strip()
        return v in ("retro","retrograde","true","1","yes")
    return SystemParams(
        Ts=f("Ts", base.Ts), rs_solar=f("rs_solar", base.rs_solar), ms_solar=f("ms_solar", base.ms_solar),
        mp_earth=f("mp_earth", base.mp_earth), dp_cgs=f("dp_cgs", base.dp_cgs),
        ap_AU=f("ap_AU", base.ap_AU), ep=f("ep", base.ep),
        mm_earth=f("mm_earth", base.mm_earth), am_hill=f("am_hill", base.am_hill), em=f("em", base.em),
        moon_retrograde=b("moon_retrograde") or b("moon_dir"),
    )

def _num(val):
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', val.strip())
        if m:
            try:
                return float(m.group(0))
            except Exception:
                return None
    return None

def _normalize_param_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    if not d:
        return {}
    out = {}
    # canonical keys
    mapping = {
        "Ts": ["Ts", "ts", "star_temp", "stellar_temp"],
        "rs_solar": ["rs_solar", "R_sun", "star_radius"],
        "ms_solar": ["ms_solar", "M_sun", "star_mass"],
        "mp_earth": ["mp_earth", "planet_mass", "M_earth_p", "mp"],
        "dp_cgs": ["dp_cgs", "planet_density", "rho_p"],
        "ap_AU": ["ap_AU", "a_planet", "semi_major_axis"],
        "ep": ["ep", "e_planet", "ecc_p"],
        # moon
        "mm_earth": ["mm_earth", "moon_mass", "M_earth_m", "mm"],
        "am_hill": ["am_hill", "a_moon_hill", "a_moon_frac", "am"],
        "em": ["em", "e_moon", "ecc_m"],
        "moon_retrograde": ["moon_retrograde","moon_dir","retrograde","orbit_dir"],
        # simulation duration
        "years": ["years","t_years","duration","sim_years"]
    }
    inv = {alias: key for key, aliases in mapping.items() for alias in aliases}
    for k, v in d.items():
        key = inv.get(k, k)
        if key == "moon_retrograde":
            val = str(v).lower().strip()
            out[key] = val in ("retro","retrograde","true","1","yes")
            continue

        if key in ("Ts","rs_solar","ms_solar","mp_earth","dp_cgs","ap_AU","ep","mm_earth","am_hill","em","years"):
            num = _num(v)
            if num is not None:
                out[key] = num
    return out

def _coerce_params_to_dict(params: Any) -> Dict[str, Any]:
    # Accept dict or JSON string
    if params is None:
        return {}
    if isinstance(params, dict):
        return params
    if isinstance(params, str):
        try:
            data = json.loads(params)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}

@mcp.tool()
def env_info() -> Dict[str, Any]:
    """Debug: interpreter and module resolution."""
    try:
        import exomoon.moon_stability as ms
        return {
            "executable": sys.executable,
            "moon_stability_file": getattr(ms, "__file__", None),
            "has_assess": hasattr(ms, "assess_moon_stability"),
            "sys_path": sys.path,
        }
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}

@mcp.tool()
def fetch_exoplanet(name: str) -> Dict[str, Any]:
    name = (name or "").strip()
    if not name:
        return {"ok": False, "message": "Empty planet name.", "data": None, "candidates": []}
    try:
        rec = fetch_system_by_planet(name)
        if rec:
            return {"ok": True, "message": f"Loaded {rec['pl_name']}", "data": rec, "candidates": []}
        cands = search_planets(name, limit=6)
        return {"ok": False, "message": f"No exact match for '{name}'.", "data": None, "candidates": cands}
    except Exception as e:
        return {"ok": False, "message": f"Error: {e}", "data": None, "candidates": []}

@mcp.tool()
def run_sim(params: Dict[str, Any]) -> Dict[str, Any]:
    try:
        p = _params_from_dict(params or {})
        t0 = perf_counter()
        sim = run_simulation(p)
        t1 = perf_counter()
        fig = build_animation(sim["traj"], sim["a_inner_au"], sim["a_outer_au"],
                              open_in_browser=False, dt=sim["dt"], t_end=sim["t_end"])
        outdir = Path("outputs"); outdir.mkdir(exist_ok=True)
        outfile = outdir / "exomoon_sim.html"
        fig.write_html(str(outfile), include_plotlyjs="cdn")
        return {
            "ok": True,
            "figure_path": str(outfile.resolve()),
            "t_end": sim["t_end"],
            "rhill_AU": sim["state"].get("rhill_AU"),
            "dt": sim["dt"],
            "n_steps": len(sim["traj"]["xyzarr_mp"]),
            "runtime_s": t1 - t0,
        }
    except Exception as e:
        print("[exomoon] run_sim error:\n", traceback.format_exc(), file=sys.stderr, flush=True)
        return {"ok": False, "message": str(e)}

@mcp.tool()
def run_sim_years(params: Dict[str, Any], years: float | None = None) -> Dict[str, Any]:
    """
    Run multi-year simulation.
    You can pass years either as the separate argument or inside params (e.g. {"years": 10}).
    Returns: figure_path, t_end, rhill_AU, dt, n_steps, runtime_s, used_years.
    """
    try:
        raw = params or {}
        # Extract years if positional not supplied
        if years is None:
            norm = _normalize_param_keys(raw)
            years = norm.get("years")
        if years is None:
            return {"ok": False, "message": "Missing years (pass argument or include 'years' in params)."}
        years_f = float(years)
        p = _params_from_dict(raw)
        t0 = perf_counter()
        sim = run_simulation_for_years(p, years_f)
        t1 = perf_counter()
        fig = build_animation(sim["traj"], sim["a_inner_au"], sim["a_outer_au"],
                              open_in_browser=False, dt=sim["dt"], t_end=sim["t_end"])
        outdir = Path("outputs"); outdir.mkdir(exist_ok=True)
        outfile = outdir / f"exomoon_sim_{int(round(years_f))}y.html"
        fig.write_html(str(outfile), include_plotlyjs="cdn")
        return {
            "ok": True,
            "figure_path": str(outfile.resolve()),
            "t_end": sim["t_end"],
            "rhill_AU": sim["state"].get("rhill_AU"),
            "dt": sim["dt"],
            "n_steps": len(sim["traj"]["xyzarr_mp"]),
            "runtime_s": t1 - t0,
            "used_years": years_f,
        }
    except Exception as e:
        print("[exomoon] run_sim_years error:\n", traceback.format_exc(), file=sys.stderr, flush=True)
        return {"ok": False, "message": str(e)}

# Keep existing names, but add the exact alias Claude mentions; all wrap errors clearly.
@mcp.tool()
def check_moon_stability(params: Dict[str, Any], years: float, escape_factor: float = 1.0) -> Dict[str, Any]:
    try:
        p = _params_from_dict(params or {})
        res = _assess_moon_stability(p, float(years), float(escape_factor))
        res["ok"] = True
        return res
    except Exception as e:
        print("[exomoon] check_moon_stability error:\n", traceback.format_exc(), file=sys.stderr, flush=True)
        return {"ok": False, "message": str(e)}

@mcp.tool()
def assess_stability(params: Dict[str, Any], years: float, escape_factor: float = 1.0) -> Dict[str, Any]:
    try:
        p = _params_from_dict(params or {})
        res = _assess_moon_stability(p, float(years), float(escape_factor))
        res["ok"] = True
        return res
    except Exception as e:
        print("[exomoon] assess_stability error:\n", traceback.format_exc(), file=sys.stderr, flush=True)
        return {"ok": False, "message": str(e)}

@mcp.tool()
def assess_moon_stability(params: Dict[str, Any], years: float, escape_factor: float = 1.0) -> Dict[str, Any]:
    """Alias matching the function name to avoid any naming confusion."""
    try:
        p = _params_from_dict(params or {})
        res = _assess_moon_stability(p, float(years), float(escape_factor))
        res["ok"] = True
        return res
    except Exception as e:
        print("[exomoon] assess_moon_stability tool error:\n", traceback.format_exc(), file=sys.stderr, flush=True)
        return {"ok": False, "message": str(e)}
    
@mcp.tool()
def moon_escape_info(params: Dict[str, Any], years: float, escape_factor: float = 1.0) -> Dict[str, Any]:
    """
    Return first escape time (years) if moon exits escape_factor * Hill radius.
    Fields: stable, escape_time (or null), escape_index, threshold, rhill_AU, max_r_rel, dt, t_end.
    """
    try:
        p = _params_from_dict(params or {})
        res = _analyze_moon_escape(p, float(years), float(escape_factor))
        res["ok"] = True
        return res
    except Exception as e:
        print("[exomoon] moon_escape_info error:\n", traceback.format_exc(), file=sys.stderr, flush=True)
        return {"ok": False, "message": str(e)}

# Update dash_url to include years in query string if provided:
@mcp.tool()
def dash_url(params: Any = None,
             planet: str | None = None,
             autorun: bool = False,
             base: str = "http://127.0.0.1:8050/") -> Dict[str, str]:
    """
    Build a Dash URL with query parameters.
    Accepts:
      params: dict or JSON string of simulation/exoplanet keys
      planet: explicit planet name (overrides params["pl_name"])
      autorun: add run=1 so Dash auto-starts simulation
      base: override base URL if running on different host/port

    Returns: url, query, accepted_keys, has_planet
    """
    try:
        raw = _coerce_params_to_dict(params)
        # Allow direct planet override
        planet_name = planet or raw.get("pl") or raw.get("pl_name") or raw.get("planet") or raw.get("name")
        norm = _normalize_param_keys(raw)

        q: Dict[str, Any] = {}
        if planet_name:
            q["pl"] = str(planet_name).strip()

        # Moon retrograde or direction
        if "moon_retrograde" in norm:
            q["moon_dir"] = "retro" if norm["moon_retrograde"] else "pro"
        else:
            # Accept raw moon_dir text
            md = raw.get("moon_dir")
            if md is not None:
                mdv = str(md).lower().strip()
                q["moon_dir"] = "retro" if mdv in ("retro","retrograde","r","true","1","yes") else "pro"

        # Numeric params
        for key in ("Ts","rs_solar","ms_solar","mp_earth","dp_cgs","ap_AU","ep",
                    "mm_earth","am_hill","em","years"):
            if key in norm:
                val = norm[key]
                if isinstance(val, (int,float)):
                    # Compact float formatting
                    if isinstance(val, float):
                        sval = f"{val:.12g}"
                    else:
                        sval = str(val)
                    if key == "years":
                        q["years"] = sval
                    else:
                        q[key] = sval

        if autorun:
            q["run"] = "1"

        query = urlencode(q)
        url = base.rstrip("/") + "/?" + query

        return {
            "ok": "true",
            "url": url,
            "query": query,
            "accepted_keys": ",".join(sorted(norm.keys())),
            "has_planet": "yes" if "pl" in q else "no",
        }
    except Exception as e:
        return {
            "ok": "false",
            "message": str(e),
            "query": "",
            "url": "",
            "accepted_keys": "",
            "has_planet": "no",
        }

if __name__ == "__main__":
    mcp.run()