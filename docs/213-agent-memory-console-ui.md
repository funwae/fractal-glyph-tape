# Agent Memory Console UI — Developer & Demo Frontend

**Goal:** Provide a **visual console** that lets you:

- Interact with an agent that uses FGMS
- Inspect the agent's memory in real time
- Drill into glyphs, clusters, and addresses
- Show off FGT as a *live* memory system

---

## 1. Overview

The Memory Console is a Next.js web application that provides:

- **Chat interface** with memory-enabled agent
- **Context visualization** showing retrieved memory items
- **Timeline view** of memory operations
- **Glyph explorer** for browsing semantic clusters
- **Address inspector** for fractal address details

Access at: `http://localhost:3000/memory-console`

---

## 2. Implementation Status

✅ **Implemented Components:**

- `MemoryChatPanel` — Chat interface with agent
- `MemoryContextPanel` — Context item visualization
- `MemoryTimeline` — Memory event timeline
- `GlyphClusterList` — Glyph browsing interface
- `AddressInspector` — Address details viewer

---

## 3. Architecture

### 3.1 Frontend (Next.js)

```
web/
  app/
    memory-console/page.tsx     # Main console page
    api/
      memory/                    # Proxy routes to Python backend
      agent/complete/route.ts    # Agent LLM endpoint
  components/
    MemoryChatPanel.tsx
    MemoryContextPanel.tsx
    MemoryTimeline.tsx
    GlyphClusterList.tsx
    AddressInspector.tsx
  types/
    memory.ts                    # TypeScript types
```

### 3.2 Backend (Python FastAPI)

```
src/memory/
  api.py          # FastAPI routes
  service.py      # Service orchestration
  store.py        # Data persistence
  policy.py       # Memory policy
```

---

## 4. Usage

### 4.1 Starting the System

1. Start Python memory server:
```bash
python scripts/run_memory_server.py
```

2. Start Next.js frontend:
```bash
cd web
npm run dev
```

3. Open browser to: `http://localhost:3000/memory-console`

### 4.2 Demo Flow

1. Enter your name as Actor ID
2. Start chatting with the agent
3. Watch the Context panel populate with memory items
4. Explore the Timeline to see memory operations
5. Browse Glyphs to see semantic clusters
6. Click addresses to inspect fractal coordinates

---

## 5. Features

### 5.1 Chat Panel

- Send messages to memory-enabled agent
- Agent retrieves relevant context from memory
- Responses incorporate past interactions
- All exchanges saved to memory

### 5.2 Context Panel

- Shows memory items retrieved for each query
- Displays glyphs, summaries, and excerpts
- Click addresses to inspect details
- See relevance scores for each item

### 5.3 Timeline

- Chronological view of memory operations
- See when memories were written
- Track region activity
- Monitor system usage

### 5.4 Glyph Explorer

- Browse semantic clusters as glyphs
- See frequency and distribution
- Understand phrase families
- Navigate fractal space

### 5.5 Address Inspector

- Decode fractal addresses
- See world, region, depth, time slice
- Understand triangular path
- Copy addresses for debugging

---

## 6. Configuration

### 6.1 Environment Variables

Create `web/.env.local`:

```env
MEMORY_API_URL=http://localhost:8001
```

### 6.2 Customization

Edit `web/app/memory-console/page.tsx`:

- Default actor ID
- Token budget range
- Available modes (glyph/text/mixed)
- UI theme and styling

---

## 7. Next Steps

- Add fractal map visualization
- Integrate glyph-drive-3d viewer
- Add export/import functionality
- Implement memory search
- Add analytics dashboard
