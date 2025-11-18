# Vercel Deployment Guide

This project is configured for deployment on Vercel with the Next.js app in the `web/` subdirectory.

## Quick Deploy

1. **Connect to Vercel:**
   - Go to [vercel.com](https://vercel.com)
   - Import your GitHub repository: `funwae/fractal-glyph-tape`
   - Vercel will auto-detect the Next.js framework

2. **Configure Project Settings:**
   - **Root Directory:** `web` (already configured in `vercel.json`)
   - **Framework Preset:** Next.js (auto-detected)
   - **Build Command:** (auto-detected, runs `npm run build` in `web/`)
   - **Output Directory:** `.next` (auto-detected)

3. **Environment Variables:**
   Add these in Vercel project settings → Environment Variables:

   - `MEMORY_API_URL` (optional): URL of the Python FastAPI memory server
     - Default: `http://localhost:8001` (for local development)
     - Production: Set to your deployed memory API URL (e.g., `https://your-memory-api.vercel.app`)
     - Note: If not set, the Memory Console features will not work, but the landing page will still function

4. **Deploy:**
   - Click "Deploy"
   - Vercel will build and deploy automatically

## Project Structure

```
.
├── vercel.json          # Vercel configuration (root directory: web/)
├── web/                 # Next.js application
│   ├── app/            # Next.js app directory
│   │   ├── api/        # API routes (serverless functions)
│   │   └── ...
│   ├── components/     # React components
│   └── package.json    # Dependencies
└── ...
```

## API Routes

The Next.js app includes API proxy routes that forward requests to the Python FastAPI backend:

- `/api/memory/read` - Memory read operations
- `/api/memory/write` - Memory write operations
- `/api/memory/addresses` - List memory addresses
- `/api/memory/regions` - List memory regions
- `/api/agent/complete` - Agent completion endpoint

These routes require the `MEMORY_API_URL` environment variable to be set.

## Features

- ✅ **Landing Page** - Works without backend
- ✅ **Explore Page** - Requires visualization API (separate deployment)
- ✅ **Memory Console** - Requires `MEMORY_API_URL` environment variable
- ✅ **API Routes** - Serverless functions on Vercel

## Troubleshooting

### Build Fails
- Ensure Node.js version is 18+ (configure in Vercel project settings)
- Check that all dependencies are in `web/package.json`

### API Routes Not Working
- Verify `MEMORY_API_URL` is set in Vercel environment variables
- Check that the Python memory API is accessible from Vercel's servers
- Review function logs in Vercel dashboard

### Static Export Issues
- The project uses serverless functions (not static export) to support API routes
- If you need static export, remove API routes or deploy them separately

## Custom Domain

After deployment, you can add a custom domain in Vercel project settings:
- Go to Settings → Domains
- Add your domain (e.g., `fgt.glyphd.com`)
- Follow DNS configuration instructions

## Environment-Specific Deployments

Vercel supports different environment variables for:
- **Production** - Main branch deployments
- **Preview** - Pull request deployments
- **Development** - Local development

Set `MEMORY_API_URL` appropriately for each environment.

