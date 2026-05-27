'use client';
import React, { useState } from 'react';
import { ExternalLink, ChevronDown, ChevronUp } from 'lucide-react';

interface AnimationHtmlEmbedProps {
  url: string;
}

export function AnimationHtmlEmbed({ url }: AnimationHtmlEmbedProps) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="border border-gray-700 rounded-lg overflow-hidden bg-gray-950">
      <button
        onClick={() => setExpanded(v => !v)}
        className="flex items-center justify-between w-full px-3 py-2 text-xs text-gray-400 hover:text-gray-200 hover:bg-gray-800/50 transition-colors"
      >
        <span className="font-semibold tracking-wide">Plotly Animation (S3)</span>
        <div className="flex items-center gap-2">
          <a
            href={url}
            target="_blank"
            rel="noreferrer"
            onClick={e => e.stopPropagation()}
            className="text-blue-400 hover:text-blue-300"
            title="Open in new tab"
          >
            <ExternalLink size={12} />
          </a>
          {expanded ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
        </div>
      </button>
      {expanded && (
        <iframe
          src={url}
          className="w-full border-t border-gray-700"
          style={{ height: 480 }}
          sandbox="allow-scripts allow-same-origin"
        />
      )}
    </div>
  );
}
