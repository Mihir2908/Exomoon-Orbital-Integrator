from pathlib import Path
from typing import Dict, Any
from urllib.parse import urlencode
import sys, traceback

from fastmcp import FastMCP

from exomoon.params import SystemParams
from exomoon.simulation import run_simulation, run_simulation_for_years
from exomoon.plotting.anim import build_animation
from exomoon.exoplanet_archive import fetch_system_by_planet, search_planets
from exomoon.moon_stability import assess_moon_stability as _assess_moon_stability  # import the function symbol

mcp = FastMCP("exomoon")

def _params_from_dict(d: Dict[str, Any]) -> SystemParams:
    base = SystemParams()
    def f(k, cur): 
        return cur if d.get(k) is None or d.get(k) == "" else float(d[k])
    return SystemParams(
        Ts=f("Ts", base.Ts), rs_solar=f("rs_solar", base.rs_solar), ms_solar=f("ms_solar", base.ms_solar),
        mp_earth=f("mp_earth", base.mp_earth), dp_cgs=f("dp_cgs", base.dp_cgs),
        ap_AU=f("ap_AU", base.ap_AU), ep=f("ep", base.ep),
        mm_earth=f("mm_earth", base.mm_earth), am_hill=f("am_hill", base.am_hill), em=f("em", base.em),
    )

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
        sim = run_simulation(p)
        fig = build_animation(sim["traj"], sim["a_inner_au"], sim["a_outer_au"], open_in_browser=False)
        outdir = Path("outputs"); outdir.mkdir(exist_ok=True)
        outfile = outdir / "exomoon_sim.html"
        fig.write_html(str(outfile), include_plotlyjs="cdn")
        return {"ok": True, "figure_path": str(outfile.resolve()), "t_end": sim["t_end"], "rhill_AU": sim["state"].get("rhill_AU")}
    except Exception as e:
        print("[exomoon] run_sim error:\n", traceback.format_exc(), file=sys.stderr, flush=True)
        return {"ok": False, "message": str(e)}

@mcp.tool()
def run_sim_years(params: Dict[str, Any], years: float) -> Dict[str, Any]:
    try:
        p = _params_from_dict(params or {})
        sim = run_simulation_for_years(p, float(years))
        fig = build_animation(sim["traj"], sim["a_inner_au"], sim["a_outer_au"], open_in_browser=False)
        outdir = Path("outputs"); outdir.mkdir(exist_ok=True)
        outfile = outdir / f"exomoon_sim_{int(years)}y.html"
        fig.write_html(str(outfile), include_plotlyjs="cdn")
        return {"ok": True, "figure_path": str(outfile.resolve()), "t_end": sim["t_end"], "rhill_AU": sim["state"].get("rhill_AU")}
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
def dash_url(params: Dict[str, Any] | None = None, planet: str | None = None, autorun: bool = False) -> Dict[str, str]:
    q = {}
    if planet: q["pl"] = planet
    if params:
        for k, v in params.items():
            if v is not None: q[k] = v
    if autorun: q["run"] = 1
    url = f"http://127.0.0.1:8050/?{urlencode(q)}" if q else "http://127.0.0.1:8050/"
    return {"url": url}

if __name__ == "__main__":
    mcp.run()