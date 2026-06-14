'use client';
import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { ChatMessage as ChatMessageType } from '@/lib/types';
import { cn } from '@/lib/utils';

interface ChatMessageProps {
  message: ChatMessageType;
}

// Detect [Download ...](url) pattern → render as <a download>
const CSV_LINK_RE = /\[Download ([^\]]+)\]\((https?:\/\/[^)]+)\)/g;

function transformCsvLinks(content: string): string {
  return content;  // Pass through — we handle in the markdown renderer
}

export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === 'user';

  return (
    <div className={cn('flex gap-2', isUser ? 'justify-end' : 'justify-start')}>
      {!isUser && (
        <div className="w-6 h-6 rounded-full bg-blue-700 flex items-center justify-center shrink-0 mt-0.5">
          <span className="text-xs text-white font-bold">A</span>
        </div>
      )}

      <div
        className={cn(
          'max-w-[85%] rounded-lg px-3 py-2 text-sm',
          isUser
            ? 'bg-blue-700 text-white rounded-br-sm'
            : 'bg-gray-800 text-gray-200 rounded-bl-sm'
        )}
      >
        {isUser ? (
          <p className="whitespace-pre-wrap">{message.content}</p>
        ) : (
          <div className="text-sm text-gray-200 leading-relaxed [&_p]:my-1 [&_ul]:list-disc [&_ul]:pl-4 [&_ol]:list-decimal [&_ol]:pl-4 [&_li]:my-0.5 [&_code]:text-blue-300 [&_code]:bg-gray-900 [&_code]:px-1 [&_code]:rounded [&_pre]:bg-gray-900 [&_pre]:p-2 [&_pre]:rounded [&_pre]:overflow-x-auto [&_pre_code]:text-xs [&_strong]:text-gray-100 [&_h1]:text-base [&_h1]:font-semibold [&_h2]:text-sm [&_h2]:font-semibold [&_h3]:text-sm [&_h3]:font-medium">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                a: ({ href, children }) => {
                  const isDownload = href && (
                    href.includes('.csv') || href.includes('.png') || href.includes('.jpg') ||
                    href.includes('animation') || href.includes('download') || href.includes('export')
                  );
                  return (
                    <a
                      href={href}
                      {...(isDownload ? { download: true } : { target: '_blank', rel: 'noreferrer' })}
                      className="text-blue-400 underline hover:text-blue-300"
                    >
                      {children}
                    </a>
                  );
                },
                img: ({ src, alt }) => (
                  <img
                    src={src}
                    alt={alt ?? 'plot'}
                    className="max-w-full rounded-md mt-2 mb-1 border border-gray-700"
                    style={{ maxHeight: '320px', objectFit: 'contain' }}
                  />
                ),
              }}
            >
              {message.content || (message.streaming ? '▋' : '')}
            </ReactMarkdown>
          </div>
        )}
      </div>

      {isUser && (
        <div className="w-6 h-6 rounded-full bg-gray-700 flex items-center justify-center shrink-0 mt-0.5">
          <span className="text-xs text-gray-300 font-bold">U</span>
        </div>
      )}
    </div>
  );
}
