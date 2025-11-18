#!/bin/bash
# Simple deployment script for Cloudflare Pages
# Usage: ./deploy.sh

set -e

echo "🚀 Deploying Fractal Glyph Tape to Cloudflare Pages..."

# Check if wrangler is installed
if ! command -v wrangler &> /dev/null; then
    echo "❌ Wrangler CLI not found. Installing..."
    npm install -g wrangler
fi

# Build the Next.js app
echo "📦 Building Next.js app..."
cd web
npm install
npm run build
cd ..

# Deploy to Cloudflare Pages
echo "☁️  Deploying to Cloudflare Pages..."
wrangler pages deploy web/out --project-name=fractal-glyph-tape

echo "✅ Deployment complete!"
echo "🌐 Your site should be live at: https://fractal-glyph-tape.pages.dev"

