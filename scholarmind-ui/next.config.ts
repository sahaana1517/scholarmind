import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Expose the API URL to the browser at build time
  env: {
    NEXT_PUBLIC_API_URL:
      process.env.NEXT_PUBLIC_API_URL ||
      "https://scholarmind-9bld.onrender.com",
  },

  // Strict mode catches subtle React bugs early
  reactStrictMode: true,

  // Compress static assets served by next start
  compress: true,

  // Strip X-Powered-By header
  poweredByHeader: false,

  // Security headers applied to every route
  async headers() {
    return [
      {
        source: "/(.*)",
        headers: [
          { key: "X-Frame-Options", value: "DENY" },
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
          {
            key: "Permissions-Policy",
            value: "camera=(), microphone=(), geolocation=()",
          },
          {
            key: "Strict-Transport-Security",
            value: "max-age=63072000; includeSubDomains; preload",
          },
        ],
      },
    ];
  },
};

export default nextConfig;
