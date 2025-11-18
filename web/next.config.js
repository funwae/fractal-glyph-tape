/** @type {import('next').NextConfig} */
const nextConfig = {
  // Remove static export to support API routes on Vercel
  // output: 'export', // Commented out for Vercel serverless functions
  images: {
    unoptimized: true,
  },
  basePath: '',
  trailingSlash: false,
}

module.exports = nextConfig
