import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Exomoon Orbital Integrator',
  description: '3-body leapfrog integrator for exomoon stability analysis',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="h-full">
      <body className="h-full overflow-hidden">{children}</body>
    </html>
  );
}
