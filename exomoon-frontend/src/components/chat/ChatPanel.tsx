'use client';
import React, { useEffect, useRef } from 'react';
import { useSimulationStore } from '@/hooks/useSimulationStore';
import { useChatStream } from '@/hooks/useChatStream';
import { ChatMessage } from './ChatMessage';
import { ChatInput } from './ChatInput';

export function ChatPanel() {
  const { chatMessages } = useSimulationStore();
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
        <div className={`ml-auto w-2 h-2 rounded-full ${isStreaming ? 'bg-blue-400 animate-pulse' : 'bg-gray-700'}`} />
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3 min-h-0">
        {chatMessages.length === 0 && (
          <div className="text-center text-gray-600 text-xs mt-8 space-y-1">
            <p>Ask questions about your simulation.</p>
            <p className="text-gray-700">e.g. "Is the moon stable?" · "Export CSV" · "What is the Hill radius?"</p>
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
