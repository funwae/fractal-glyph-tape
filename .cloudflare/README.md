# Cloudflare Pages Deployment

This project is configured for deployment to Cloudflare Pages.

## Quick Deploy

### Option 1: Using Wrangler CLI (Recommended)

1. **Install Wrangler** (if not already installed):
   ```bash
   npm install -g wrangler
   ```

2. **Login to Cloudflare**:
   ```bash
   wrangler login
   ```

3. **Create a Pages project** (first time only):
   ```bash
   wrangler pages project create fractal-glyph-tape
   ```

4. **Deploy**:
   ```bash
   cd web
   npm run build
   cd ..
   wrangler pages deploy web/out --project-name=fractal-glyph-tape
   ```

### Option 2: Using Cloudflare Dashboard

1. Go to [Cloudflare Dashboard](https://dash.cloudflare.com/) → Pages
2. Click "Create a project"
3. Connect your GitHub repository
4. Configure build settings:
   - **Framework preset**: Next.js (Static HTML Export)
   - **Build command**: `cd web && npm install && npm run build`
   - **Build output directory**: `web/out`
   - **Root directory**: `/` (project root)

## Configuration

The `wrangler.toml` file in the project root contains the deployment configuration.

## Important Notes

- **API Routes**: The Next.js app uses `output: 'export'` which creates a static site. API routes in `app/api/` will not work in the static build. These routes proxy to a Python backend that needs to be deployed separately.

- **Environment Variables**: If you need to set environment variables (like `MEMORY_API_URL`), add them in the Cloudflare Pages dashboard under Settings → Environment Variables.

- **Custom Domain**: Configure custom domains in the Cloudflare Pages dashboard after deployment.

## Build Output

The static export creates files in `web/out/` directory, which is what Cloudflare Pages serves.

