# Cloudflare Pages Deployment Guide

This project is configured for deployment to Cloudflare Pages.

## Files Created

- `wrangler.toml` - Cloudflare Pages configuration
- `.cloudflare/deploy.sh` - Simple deployment script
- `.cloudflare/README.md` - Detailed deployment instructions
- `.github/workflows/cloudflare-pages.yml` - GitHub Actions workflow (optional)

## Quick Start

### Method 1: Using Wrangler CLI (Recommended)

1. **Install Wrangler**:
   ```bash
   npm install -g wrangler
   ```

2. **Login to Cloudflare**:
   ```bash
   wrangler login
   ```

3. **Create Pages project** (first time only):
   ```bash
   wrangler pages project create fractal-glyph-tape
   ```

4. **Deploy**:
   ```bash
   ./.cloudflare/deploy.sh
   ```

   Or manually:
   ```bash
   cd web
   npm install
   npm run build
   cd ..
   wrangler pages deploy web/out --project-name=fractal-glyph-tape
   ```

### Method 2: Using Cloudflare Dashboard

1. Go to [Cloudflare Dashboard](https://dash.cloudflare.com/) → Pages → Create a project
2. Connect your GitHub repository: `funwae/fractal-glyph-tape`
3. Configure build settings:
   - **Framework preset**: Next.js (Static HTML Export)
   - **Build command**: `cd web && npm install && npm run build`
   - **Build output directory**: `web/out`
   - **Root directory**: `/` (leave as project root)
4. Click "Save and Deploy"

### Method 3: GitHub Actions (Automatic)

1. Add secrets to your GitHub repository:
   - `CLOUDFLARE_API_TOKEN` - Get from Cloudflare Dashboard → My Profile → API Tokens
   - `CLOUDFLARE_ACCOUNT_ID` - Get from Cloudflare Dashboard → Right sidebar
2. Push to `main` branch - deployment will happen automatically

## Configuration

The `wrangler.toml` file contains:
- Project name: `fractal-glyph-tape`
- Build output directory: `web/out`
- Compatibility date: `2025-11-17`

## Environment Variables

If your app needs environment variables (like `MEMORY_API_URL`):

1. Go to Cloudflare Dashboard → Pages → Your Project → Settings → Environment Variables
2. Add variables for Production, Preview, or both
3. Redeploy for changes to take effect

## Important Notes

⚠️ **API Routes**: The Next.js app uses `output: 'export'` which creates a static site. API routes in `app/api/` will **not work** in the static build. These routes proxy to a Python backend that needs to be deployed separately.

✅ **Static Pages**: All pages in `app/` (like `/`, `/demo`, `/explore`, `/memory-console`) will work perfectly as static HTML.

## Custom Domain

After deployment:
1. Go to Cloudflare Dashboard → Pages → Your Project → Custom domains
2. Click "Set up a custom domain"
3. Follow the instructions to add your domain

## Troubleshooting

- **Build fails**: Check that `web/package.json` has all dependencies
- **404 errors**: Ensure `next.config.js` has `trailingSlash: true` (already configured)
- **API routes not working**: This is expected - they need a separate backend deployment

## Support

For more information, see:
- [Cloudflare Pages Docs](https://developers.cloudflare.com/pages/)
- [Wrangler CLI Docs](https://developers.cloudflare.com/workers/wrangler/)

