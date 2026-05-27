import { NextRequest, NextResponse } from 'next/server';

const TAP = 'https://exoplanetarchive.ipac.caltech.edu/TAP/sync';

async function tapPost(sql: string): Promise<unknown[]> {
  const body = new URLSearchParams({ query: sql, format: 'json' });
  const r = await fetch(TAP, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: body.toString(),
    next: { revalidate: 300 },
  });
  if (!r.ok) return [];
  return r.json();
}

// Mirrors search_planets() in exoplanet_archive.py
export async function GET(req: NextRequest) {
  const q = req.nextUrl.searchParams.get('q') ?? '';
  if (q.length < 3) return NextResponse.json({ planets: [] });

  const qEsc = q.replace(/'/g, "''");
  const cond = `UPPER(pl_name) LIKE UPPER('%${qEsc}%')`;
  const numMatch = q.match(/\d+/);
  const order = numMatch
    ? `CASE WHEN UPPER(pl_name) LIKE UPPER('${qEsc}%') THEN 0 WHEN UPPER(pl_name) LIKE UPPER('% ${numMatch[0]}%') THEN 1 ELSE 2 END, pl_name`
    : `CASE WHEN UPPER(pl_name) LIKE UPPER('${qEsc}%') THEN 0 ELSE 1 END, pl_name`;

  const sql = `SELECT TOP 25 pl_name, hostname FROM pscomppars WHERE ${cond} ORDER BY ${order}`;

  try {
    const rows = await tapPost(sql) as Array<{ pl_name: string; hostname: string }>;
    return NextResponse.json({ planets: rows.filter(r => r.pl_name) });
  } catch {
    return NextResponse.json({ planets: [] });
  }
}
