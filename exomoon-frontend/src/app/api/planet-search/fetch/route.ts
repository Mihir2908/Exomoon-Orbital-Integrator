import { NextRequest, NextResponse } from 'next/server';

const TAP = 'https://exoplanetarchive.ipac.caltech.edu/TAP/sync';
const COLS = 'pl_name,hostname,st_teff,st_rad,st_mass,pl_bmasse,pl_rade,pl_orbsmax,pl_orbeccen';

async function tapPost(sql: string): Promise<unknown[]> {
  const body = new URLSearchParams({ query: sql, format: 'json' });
  const r = await fetch(TAP, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: body.toString(),
    next: { revalidate: 3600 },
  });
  if (!r.ok) throw new Error(`TAP ${r.status}`);
  return r.json();
}

// Mirrors fetch_system_by_planet() in exoplanet_archive.py
export async function POST(req: NextRequest) {
  const { name } = await req.json() as { name: string };
  if (!name) return NextResponse.json({ ok: false });

  const lit = name.trim().replace(/'/g, "''");

  // Try exact match first, then case-insensitive, then LIKE (mirrors Python fallback chain)
  let rows: unknown[] = [];
  try { rows = await tapPost(`SELECT ${COLS} FROM pscomppars WHERE pl_name='${lit}'`); } catch { rows = []; }
  if (!rows.length) {
    try { rows = await tapPost(`SELECT ${COLS} FROM pscomppars WHERE UPPER(pl_name)=UPPER('${lit}')`); } catch { rows = []; }
  }
  if (!rows.length) {
    try { rows = await tapPost(`SELECT TOP 1 ${COLS} FROM pscomppars WHERE UPPER(pl_name) LIKE UPPER('%${lit}%') ORDER BY pl_name`); } catch { rows = []; }
  }

  if (!rows.length) return NextResponse.json({ ok: false });

  const row = rows[0] as Record<string, number | string | null>;
  return NextResponse.json({
    ok: true,
    record: {
      pl_name:  row.pl_name,
      hostname: row.hostname,
      Ts:       row.st_teff   ?? null,
      rs_solar: row.st_rad    ?? null,
      ms_solar: row.st_mass   ?? null,
      mp_earth: row.pl_bmasse ?? null,  // pl_bmasse matches Python (best mass)
      pl_rade:  row.pl_rade   ?? null,
      ap_AU:    row.pl_orbsmax   ?? null,
      ep:       row.pl_orbeccen  ?? null,
    },
  });
}
