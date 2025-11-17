# Fractal Glyph Tape - Web Interface

Next.js web application for Fractal Glyph Tape, including the Memory Console UI.

## Features

- **Landing Page**: Overview and introduction to FGT
- **Memory Console**: Interactive agent memory interface (`/memory-console`)
- **API Proxy**: Routes to Python FastAPI backend

## Getting Started

### Prerequisites

- Node.js 18+
- Python memory server running (see root README)

### Installation

```bash
npm install
```

### Development

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

### Memory Console

Navigate to [http://localhost:3000/memory-console](http://localhost:3000/memory-console)

Requirements:
- Python memory server must be running at `http://localhost:8001`
- Configure `MEMORY_API_URL` in `.env.local` if using different URL

### Build

```bash
npm run build
npm start
```

## Project Structure

```
app/
  page.tsx                    # Landing page
  memory-console/page.tsx     # Memory Console
  api/                        # API routes
    memory/                   # Memory API proxies
    agent/                    # Agent backend
components/
  MemoryChatPanel.tsx        # Chat interface
  MemoryContextPanel.tsx     # Context visualization
  MemoryTimeline.tsx         # Event timeline
  GlyphClusterList.tsx       # Glyph explorer
  AddressInspector.tsx       # Address details
types/
  memory.ts                   # TypeScript types
```

## Environment Variables

Copy `.env.local.example` to `.env.local`:

```bash
cp .env.local.example .env.local
```

Required variables:
- `MEMORY_API_URL`: URL of Python memory server (default: `http://localhost:8001`)

## Learn More

- [Phase 4 Documentation](../docs/210-phase-4-agent-memory-and-api.md)
- [Memory Console UI Guide](../docs/213-agent-memory-console-ui.md)
- [Evaluation & Testing](../docs/211-agent-memory-eval-and-test-plan.md)
