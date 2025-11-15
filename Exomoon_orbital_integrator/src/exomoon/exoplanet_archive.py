import requests
import re

API = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

COLS = ",".join([
    "pl_name","hostname","st_teff","st_rad","st_mass",
    "pl_bmasse","pl_rade","pl_orbsmax","pl_orbeccen",
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

def search_planets(query: str, limit: int = 25) -> list[str]:
    """
    Return up to 'limit' planet names matching query, ranked so:
      1) Names starting with the typed text (prefix) first
      2) If a numeric chunk is present (e.g., '147'), names containing ' 147' next
      3) Then other substring matches
    """
    if not query:
        return []
    # Normalize spacing and escape
    q = " ".join(query.split())
    q_esc = q.replace("'", "''")

    # Build ordering that prioritizes prefix and then numeric chunk (if any)
    m = re.search(r"\d+", q)
    if m:
        num = m.group(0).replace("'", "''")
        order = (
            f"CASE "
            f"WHEN UPPER(pl_name) LIKE UPPER('{q_esc}%') THEN 0 "            # starts with typed text
            f"WHEN UPPER(pl_name) LIKE UPPER('% {num}%') THEN 1 "             # contains number token
            f"ELSE 2 END, pl_name"
        )
    else:
        order = (
            f"CASE "
            f"WHEN UPPER(pl_name) LIKE UPPER('{q_esc}%') THEN 0 "
            f"ELSE 1 END, pl_name"
        )

    cond = f"UPPER(pl_name) LIKE UPPER('%{q_esc}%')"
    sql = f"SELECT TOP {int(limit)} pl_name FROM pscomppars WHERE {cond} ORDER BY {order}"
    rows = _query_sql(sql)
    return [r.get("pl_name") for r in rows if r.get("pl_name")]
    
def estimate_density_gcc(mp_earth: float | None, pr_earth: float | None) -> float | None:
    if mp_earth is None or pr_earth is None or pr_earth <= 0:
        return None
    return 5.514 * (mp_earth / (pr_earth**3))