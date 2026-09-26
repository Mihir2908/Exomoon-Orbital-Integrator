'use client';
import React, { useEffect, useRef } from 'react';
import { useSimulationStore } from '@/hooks/useSimulationStore';
import { useChatStream } from '@/hooks/useChatStream';
import { ChatMessage } from './ChatMessage';
import { ChatInput } from './ChatInput';

export function ChatPanel() {
  const { chatMessages, clearSession } = useSimulationStore();
  const { sendMessage } = useChatStream();
  const bottomRef = useRef<HTMLDivElement>(null);
  const isStreaming = chatMessages.some(m => m.streaming);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages]);

  return (
    <div className="flex flex-col h-full bg-gray-950">
      {/* Header */}
      <div className="flex items-center px-4 py-3 border-b border-gray-800 shrink-0">
        <h2 className="text-xs font-semibold text-gray-400 tracking-widest uppercase">Agent Chat</h2>
        <button
          onClick={clearSession}
          disabled={isStreaming}
          title="Start a new session — clears chat history and cached simulation data"
          className="ml-auto mr-2 px-2 py-0.5 rounded text-[10px] font-medium text-gray-500
                     border border-gray-700 hover:border-gray-500 hover:text-gray-300
                     disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
        >
          New Session
        </button>
        <div className={`w-2 h-2 rounded-full ${isStreaming ? 'bg-blue-400 animate-pulse' : 'bg-gray-700'}`} />
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3 min-h-0">
        {chatMessages.length === 0 && (
          <div className="mt-4 space-y-3">
            <p className="text-center text-gray-500 text-[11px] px-2">
              Ask me anything about the simulation, or try one of these:
            </p>
            {([
              { text: "Is the moon in the current configuration stable?",         note: "Stability analysis" },
              { text: "Run an ML stability grid to find viable moon configurations", note: "ML Layer 1" },
              { text: "Fetch parameters for Kepler-452b and set up the system",   note: "NASA archive" },
              { text: "Show a trajectory preview of stable and habitable moon orbits", note: "ML Layer 2" },
              { text: "What does the Hill radius tell us about moon stability?",   note: "Explainer" },
              { text: "What can I do with this tool?",                             note: "Overview" },
            ] as const).map(({ text, note }) => (
              <button
                key={text}
                onClick={() => sendMessage(text)}
                className="w-full text-left px-3 py-2 rounded-lg bg-gray-800/60 border border-gray-700/50
                           text-gray-400 text-[11px] hover:border-violet-600/50 hover:text-gray-200
                           hover:bg-gray-800 transition-colors group"
              >
                <span>{text}</span>
                <span className="ml-1.5 text-gray-600 group-hover:text-gray-500 text-[10px]">— {note}</span>
              </button>
            ))}
          </div>
        )}
        {chatMessages.map(msg => (
          <ChatMessage key={msg.id} message={msg} />
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <ChatInput onSend={sendMessage} disabled={isStreaming} />
    </div>
  );
}
