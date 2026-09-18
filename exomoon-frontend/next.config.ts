import type { NextConfig } from 'next';

const AGENT_BACKEND = process.env.AGENT_BACKEND_URL ?? 'http://127.0.0.1:8000';

const nextConfig: NextConfig = {
  output: 'standalone',
  // Proxy /api/agent/* → agent service so the browser never makes cross-origin requests.
  // This eliminates CORS/PNA issues in all browsers without touching agent service code.
  async rewrites() {
    return [
      {
        source: '/api/agent/:path*',
        destination: `${AGENT_BACKEND}/:path*`,
      },
    ];
  },
  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [{ key: 'X-Content-Type-Options', value: 'nosniff' }],
      },
    ];
  },
};

export default nextConfig;
