'use client';
import React, { useState } from 'react';
import { MessageSquare } from 'lucide-react';
import { cn } from '@/lib/utils';

interface AppShellProps {
  main: React.ReactNode;
  chat: React.ReactNode;
}

export function AppShell({ main, chat }: AppShellProps) {
  const [chatOpen, setChatOpen] = useState(false);

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-gray-950 text-gray-100 font-sans">
      {/* Main content area — full width */}
      <main className="flex-1 min-w-0 overflow-hidden">
        {main}
      </main>

      {/* Chat drawer — slides in from right */}
      <div
        className={cn(
          'fixed top-0 right-0 h-full w-96 bg-gray-900 border-l border-gray-800 shadow-2xl z-50',
          'flex flex-col transition-transform duration-300 ease-in-out',
          chatOpen ? 'translate-x-0' : 'translate-x-full'
        )}
      >
        {chat}
      </div>

      {/* Chat toggle FAB */}
      <button
        onClick={() => setChatOpen(o => !o)}
        className={cn(
          'fixed bottom-5 right-4 z-50 w-12 h-12 rounded-full shadow-lg',
          'flex items-center justify-center transition-colors',
          chatOpen
            ? 'bg-gray-700 hover:bg-gray-600'
            : 'bg-blue-600 hover:bg-blue-500'
        )}
        title="Toggle chat"
      >
        <MessageSquare size={20} />
      </button>
    </div>
  );
}
