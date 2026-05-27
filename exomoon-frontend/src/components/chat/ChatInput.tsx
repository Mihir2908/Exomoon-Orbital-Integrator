'use client';
import React, { useState, useRef } from 'react';
import { Send } from 'lucide-react';
import { cn } from '@/lib/utils';

interface ChatInputProps {
  onSend: (msg: string) => void;
  disabled?: boolean;
}

export function ChatInput({ onSend, disabled }: ChatInputProps) {
  const [value, setValue] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const submit = () => {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setValue('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  const onInput = () => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  };

  return (
    <div className="flex items-end gap-2 p-3 border-t border-gray-800 bg-gray-950">
      <textarea
        ref={textareaRef}
        value={value}
        onChange={e => setValue(e.target.value)}
        onKeyDown={onKeyDown}
        onInput={onInput}
        disabled={disabled}
        placeholder="Ask about the simulation…"
        rows={1}
        className={cn(
          'flex-1 resize-none overflow-hidden bg-gray-800 border border-gray-700 rounded-lg',
          'px-3 py-2 text-sm text-gray-200 placeholder-gray-500',
          'focus:outline-none focus:border-blue-500 transition-colors',
          disabled && 'opacity-50 cursor-not-allowed'
        )}
      />
      <button
        onClick={submit}
        disabled={disabled || !value.trim()}
        className={cn(
          'flex items-center justify-center w-8 h-8 rounded-lg transition-colors shrink-0',
          value.trim() && !disabled
            ? 'bg-blue-600 hover:bg-blue-500 text-white'
            : 'bg-gray-800 text-gray-600 cursor-not-allowed'
        )}
      >
        <Send size={14} />
      </button>
    </div>
  );
}
