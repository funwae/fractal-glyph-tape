# Fractal Glyph Tape Landing Page

This is the landing page for Fractal Glyph Tape, built with Next.js 14 and deployed on Vercel.

## Development

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build

# Start production server
npm start
```

## Deployment

This site is configured for static export and deploys to Vercel.

### Deploy to Vercel

1. Push to GitHub
2. Import project in Vercel dashboard
3. Configure:
   - **Root Directory:** `web`
   - **Build Command:** `npm run build`
   - **Output Directory:** `out`
   - **Install Command:** `npm install`

Or use Vercel CLI:

```bash
cd web
npm install -g vercel
vercel
```

## Stack

- **Framework:** Next.js 14 (App Router)
- **Styling:** Tailwind CSS
- **Icons:** Lucide React
- **Deployment:** Vercel
- **Rendering:** Static Export (SSG)

## Structure

```
web/
├── app/
│   ├── layout.tsx       # Root layout with metadata
│   ├── page.tsx         # Main landing page
│   └── globals.css      # Global styles
├── components/
│   ├── Hero.tsx         # Hero section
│   ├── Features.tsx     # Features grid
│   ├── WhyItMatters.tsx # Benefits section
│   ├── HowItWorks.tsx   # Pipeline explanation
│   ├── ForResearchers.tsx # Research resources
│   └── Footer.tsx       # Footer with links
├── public/              # Static assets
├── next.config.js       # Next.js configuration
├── tailwind.config.ts   # Tailwind configuration
└── package.json         # Dependencies
```

## License

Proprietary License - Non-commercial use only. See main repository LICENSE file for details.
