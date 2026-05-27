'use client';
import React, { useState, useCallback, useRef, useEffect } from 'react';
import { Search, Loader2 } from 'lucide-react';
import { useSimulationStore } from '@/hooks/useSimulationStore';
import { estimateDensityGcc } from '@/lib/paramDefaults';
import { cn } from '@/lib/utils';

interface Planet { pl_name: string; hostname: string }

export function NasaSearch({ hideLabel = false }: { hideLabel?: boolean } = {}) {
  const { setParams } = useSimulationStore();
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<Planet[]>([]);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState('');
  const [open, setOpen] = useState(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Close dropdown on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (!containerRef.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const search = useCallback((q: string) => {
    if (q.length < 3) { setResults([]); setOpen(false); return; }
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(async () => {
      setLoading(true);
      try {
        const r = await fetch(`/api/planet-search?q=${encodeURIComponent(q)}`);
        const data = await r.json();
        setResults(data.planets ?? []);
        setOpen(true);
      } catch { setResults([]); }
      finally { setLoading(false); }
    }, 300);
  }, []);

  const handleSelect = async (name: string) => {
    setOpen(false);
    setQuery(name);
    setStatus('Fetching…');
    try {
      const r = await fetch('/api/planet-search/fetch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name }),
      });
      const data = await r.json();
      if (data.ok && data.record) {
        const rec = data.record;
        const updates: Partial<Parameters<typeof setParams>[0]> = {};
        if (rec.Ts)       updates.Ts       = rec.Ts;
        if (rec.rs_solar) updates.rs_solar = rec.rs_solar;
        if (rec.ms_solar) updates.ms_solar = rec.ms_solar;
        if (rec.mp_earth) updates.mp_earth = rec.mp_earth;
        if (rec.ap_AU)    updates.ap_AU    = rec.ap_AU;
        if (rec.ep != null) updates.ep     = rec.ep;
        const rho = estimateDensityGcc(rec.mp_earth ?? null, rec.pl_rade ?? null);
        if (rho) updates.dp_cgs = rho;
        setParams(updates);
        setStatus(`✓ Loaded ${rec.pl_name ?? name}`);
      } else {
        setStatus('No data found');
      }
    } catch {
      setStatus('Fetch error');
    }
  };

  return (
    <div ref={containerRef} className="relative space-y-1">
      {!hideLabel && <label className="text-xs text-gray-400">NASA Exoplanet Archive</label>}
      <div className="relative">
        <Search size={14} className="absolute left-2.5 top-2.5 text-gray-500" />
        <input
          value={query}
          onChange={e => { setQuery(e.target.value); search(e.target.value); }}
          placeholder="Type 3+ chars to search…"
          className={cn(
            'w-full pl-8 pr-3 py-1.5 text-xs rounded bg-gray-800 border border-gray-700',
            'text-gray-200 placeholder-gray-600 focus:outline-none focus:border-blue-500'
          )}
        />
        {loading && <Loader2 size={12} className="absolute right-2.5 top-2.5 text-gray-500 animate-spin" />}
      </div>
      {open && results.length > 0 && (
        <ul className="absolute z-50 w-full mt-1 max-h-48 overflow-y-auto bg-gray-800 border border-gray-700 rounded shadow-xl">
          {results.map(p => (
            <li
              key={p.pl_name}
              onMouseDown={() => handleSelect(p.pl_name)}
              className="px-3 py-1.5 text-xs cursor-pointer hover:bg-gray-700 text-gray-200"
            >
              <span className="font-medium">{p.pl_name}</span>
              <span className="text-gray-500 ml-2">{p.hostname}</span>
            </li>
          ))}
        </ul>
      )}
      {status && <p className="text-xs text-blue-400">{status}</p>}
    </div>
  );
}
