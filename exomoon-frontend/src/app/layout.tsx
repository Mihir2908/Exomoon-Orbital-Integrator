import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Planetary Orbital Dynamics Lab',
  description: '3-body orbital dynamics simulator with ML stability prediction and AI chatbot',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="h-full">
      <body className="h-full overflow-hidden">{children}</body>
    </html>
  );
}
