import io, json, base64
import numpy as np

try:
    import pandas as pd
    _HAS_PANDAS = True
except Exception:
    _HAS_PANDAS = False

def traj_to_frame(sim: dict):
    traj = sim["traj"]
    xyz_mp = traj["xyzarr_mp"]
    xyz_ms = traj["xyzarr_ms"]
    xyz_mm = traj["xyzarr_mm"]
    vel_mp = traj.get("velarr_mp")
    vel_ms = traj.get("velarr_ms")
    vel_mm = traj.get("velarr_mm")

    dt = sim["dt"]
    n = xyz_mp.shape[0]
    t = np.arange(n) * dt  # years

    # Relative vectors and distances
    rel_mm_mp = xyz_mm - xyz_mp
    rel_mp_ms = xyz_mp - xyz_ms
    moon_planet_dist = np.sqrt((rel_mm_mp**2).sum(axis=1))
    planet_star_dist = np.sqrt((rel_mp_ms**2).sum(axis=1))

    data = {
        "t_years": t,
        # positions
        "star_x": xyz_ms[:, 0], "star_y": xyz_ms[:, 1], "star_z": xyz_ms[:, 2],
        "planet_x": xyz_mp[:, 0], "planet_y": xyz_mp[:, 1], "planet_z": xyz_mp[:, 2],
        "moon_x": xyz_mm[:, 0], "moon_y": xyz_mm[:, 1], "moon_z": xyz_mm[:, 2],
        # relative pos + distances
        "moon_rel_x": rel_mm_mp[:, 0], "moon_rel_y": rel_mm_mp[:, 1], "moon_rel_z": rel_mm_mp[:, 2],
        "planet_rel_x": rel_mp_ms[:, 0], "planet_rel_y": rel_mp_ms[:, 1], "planet_rel_z": rel_mp_ms[:, 2],
        "moon_planet_dist": moon_planet_dist,
        "planet_star_dist": planet_star_dist,
    }

    # velocities (if present)
    if vel_mp is not None and vel_ms is not None and vel_mm is not None:
        data.update({
            "star_vx": vel_ms[:, 0], "star_vy": vel_ms[:, 1], "star_vz": vel_ms[:, 2],
            "planet_vx": vel_mp[:, 0], "planet_vy": vel_mp[:, 1], "planet_vz": vel_mp[:, 2],
            "moon_vx": vel_mm[:, 0], "moon_vy": vel_mm[:, 1], "moon_vz": vel_mm[:, 2],
            "moon_speed": np.sqrt((vel_mm**2).sum(axis=1)),
            "planet_speed": np.sqrt((vel_mp**2).sum(axis=1)),
            "star_speed": np.sqrt((vel_ms**2).sum(axis=1)),
        })

    if _HAS_PANDAS:
        return pd.DataFrame(data)
    return data  # dict-of-arrays fallback

def to_csv_bytes(frame) -> bytes:
    if _HAS_PANDAS:
        buf = io.StringIO()
        frame.to_csv(buf, index=False)
        return buf.getvalue().encode()
    # Manual CSV for dict-of-arrays
    cols = list(frame.keys())
    n = len(frame[cols[0]])
    out = io.StringIO()
    out.write(",".join(cols) + "\n")
    for i in range(n):
        out.write(",".join(str(frame[c][i]) for c in cols) + "\n")
    return out.getvalue().encode()

def _pretty_name(name: str) -> str:
    # Human-friendly labels
    if name == "t_years": return "Time"
    if name == "moon_planet_dist": return "Moon–Planet Distance"
    if name == "planet_star_dist": return "Planet–Star Distance"
    if name.endswith("_speed"):
        body = name.replace("_speed", "").capitalize()
        return f"{body} Speed"
    # Components
    if name.endswith(("_x", "_y", "_z")):
        base, comp = name.rsplit("_", 1)
        return f"{base.replace('_', ' ').title()} {comp.upper()}"
    return name.replace("_", " ").title()

def _unit_for(name: str) -> str | None:
    # All state is in AU and years (G = 4π² convention)
    if name == "t_years": return "years"
    if name in ("moon_planet_dist", "planet_star_dist"): return "AU"
    if name.endswith(("_x", "_y", "_z")): return "AU"
    if name.endswith(("_vx", "_vy", "_vz")): return "AU/yr"
    if name.endswith("_speed"): return "AU/yr"
    if name.startswith(("moon_rel_", "planet_rel_")): return "AU"
    return None

def var_info(name: str) -> tuple[str, str | None]:
    """
    Return (pretty_label, unit or None) for a variable.
    """
    return _pretty_name(name), _unit_for(name)


def pack_sim(sim: dict) -> str:
    # Store minimal arrays + dt + t_end; include velocities if present; base64 for compact transport
    traj = sim["traj"]
    payload = {
        "dt": sim["dt"],
        "t_end": sim["t_end"],
        "xyz_mp": traj["xyzarr_mp"].tolist(),
        "xyz_ms": traj["xyzarr_ms"].tolist(),
        "xyz_mm": traj["xyzarr_mm"].tolist(),
        "vel_mp": traj.get("velarr_mp").tolist() if traj.get("velarr_mp") is not None else None,
        "vel_ms": traj.get("velarr_ms").tolist() if traj.get("velarr_ms") is not None else None,
        "vel_mm": traj.get("velarr_mm").tolist() if traj.get("velarr_mm") is not None else None,
    }
    raw = json.dumps(payload).encode()
    return base64.b64encode(raw).decode()

def unpack_sim(packed: str) -> dict:
    raw = base64.b64decode(packed.encode())
    payload = json.loads(raw.decode())
    traj = {
        "xyzarr_mp": np.array(payload["xyz_mp"], dtype=float),
        "xyzarr_ms": np.array(payload["xyz_ms"], dtype=float),
        "xyzarr_mm": np.array(payload["xyz_mm"], dtype=float),
        "velarr_mp": np.array(payload["vel_mp"], dtype=float) if payload.get("vel_mp") is not None else None,
        "velarr_ms": np.array(payload["vel_ms"], dtype=float) if payload.get("vel_ms") is not None else None,
        "velarr_mm": np.array(payload["vel_mm"], dtype=float) if payload.get("vel_mm") is not None else None,
    }
    return {"dt": payload["dt"], "t_end": payload["t_end"], "traj": traj}