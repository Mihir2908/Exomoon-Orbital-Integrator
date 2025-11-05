import requests

API = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

COLS = ",".join([
    "pl_name",
    "hostname",
    "st_teff",
    "st_rad",
    "st_mass",
    "pl_bmasse",
    "pl_rade",
    "pl_orbsmax",
    "pl_orbeccen",
])

HEADERS = {"User-Agent": "ExomoonOrbitalIntegrator/0.1 (contact: your-email@example.com)"}

def _query_sql(sql: str):
    resp = requests.post(API, data={"query": sql, "format": "json"}, headers=HEADERS, timeout=20)
    if not resp.ok:
        msg = resp.text.strip()[:500]
        raise requests.HTTPError(f"TAP error {resp.status_code}: {msg}")
    return resp.json()

def fetch_system_by_planet(pl_name: str) -> dict | None:
    if not pl_name:
        return None
    name = pl_name.strip()
    name_lit = name.replace("'", "''")

    # pscomppars already returns a single composite solution per planet
    sql1 = f"SELECT {COLS} FROM pscomppars WHERE pl_name='{name_lit}'"
    try:
        rows = _query_sql(sql1)
    except requests.HTTPError:
        rows = []

    if not rows:
        sql2 = f"SELECT {COLS} FROM pscomppars WHERE UPPER(pl_name)=UPPER('{name_lit}')"
        try:
            rows = _query_sql(sql2)
        except requests.HTTPError:
            rows = []

    if not rows:
        like_lit = f"%{name_lit}%"
        sql3 = f"SELECT TOP 1 {COLS} FROM pscomppars WHERE UPPER(pl_name) LIKE UPPER('{like_lit}') ORDER BY pl_name"
        rows = _query_sql(sql3)

    if not rows:
        return None

    row = rows[0]
    return {
        "pl_name": row.get("pl_name"),
        "hostname": row.get("hostname"),
        "Ts": row.get("st_teff"),
        "rs_solar": row.get("st_rad"),
        "ms_solar": row.get("st_mass"),
        "mp_earth": row.get("pl_bmasse"),
        "pl_rade": row.get("pl_rade"),
        "ap_AU": row.get("pl_orbsmax"),
        "ep": row.get("pl_orbeccen"),
    }

def estimate_density_gcc(mp_earth: float | None, pr_earth: float | None) -> float | None:
    if mp_earth is None or pr_earth is None or pr_earth <= 0:
        return None
    return 5.514 * (mp_earth / (pr_earth**3))