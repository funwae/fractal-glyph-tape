# Phase 4 Quick Start Guide

Get the Fractal Glyph Memory System running in 5 minutes.

## Prerequisites

- Python 3.10+
- Node.js 18+
- Terminal access

## Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Start the Memory API

```bash
python scripts/start_memory_api.py --reload
```

You should see:
```
Starting Fractal Glyph Memory System API on 0.0.0.0:8001
API documentation: http://localhost:8001/docs
Memory Console: http://localhost:3000/memory-console
```

## Step 3: Test the API (New Terminal)

```bash
python scripts/test_memory_api.py
```

Expected output:
```
=== Test 1: Write Memory ===
✓ Memory written successfully

=== Test 2: Read Memory ===
✓ Memory read successfully

=== Test 3: Stats ===
✓ Stats retrieved successfully

=== Test 4: Agent Chat ===
✓ Agent chat successful

✓ All tests passed!
```

## Step 4: Try Manual API Calls

### Write a memory:
```bash
curl -X POST http://localhost:8001/api/memory/write \
  -H "Content-Type: application/json" \
  -d '{
    "actor_id": "demo_user",
    "text": "Today I started using fractal memory for my AI projects.",
    "tags": ["devlog", "ai"],
    "source": "user"
  }'
```

### Read memories:
```bash
curl -X POST http://localhost:8001/api/memory/read \
  -H "Content-Type: application/json" \
  -d '{
    "actor_id": "demo_user",
    "query": "What did I do today?",
    "token_budget": 2048,
    "mode": "mixed"
  }'
```

## Step 5: Launch Memory Console (New Terminal)

```bash
cd web
npm install  # First time only
npm run dev
```

Visit: http://localhost:3000/memory-console

Try:
1. Click "Test Write" to add sample memory
2. Click "Test Read" to retrieve it
3. Type in chat: "What did I work on today?"
4. Watch the context panel update with memories

## Step 6: Run Benchmarks

### Storage Compression:
```bash
python scripts/bench_storage.py --num 100
```

Expected: ~0.55x compression ratio

### Context Efficiency:
```bash
python scripts/bench_context_efficiency.py
```

Expected: 20-40% improvement over baseline

Results saved to `reports/` directory.

## What to Try Next

### Experiment with Policies

Change `policyMode` in the console:
- **recent**: Most recent memories first
- **relevant**: Most relevant to query
- **mixed**: Best of both (default)

### Test Multi-Actor Isolation

```bash
# Actor 1
curl -X POST http://localhost:8001/api/memory/write \
  -d '{"actor_id":"alice","text":"I prefer React","tags":["tech"]}'

# Actor 2
curl -X POST http://localhost:8001/api/memory/write \
  -d '{"actor_id":"bob","text":"I prefer Vue","tags":["tech"]}'

# Read - Alice's memories only
curl -X POST http://localhost:8001/api/memory/read \
  -d '{"actor_id":"alice","query":"frontend framework","mode":"relevant"}'
```

### Explore Fractal Addresses

Memories are automatically assigned addresses like:
```
earth/tech/012210/6/2025-11-17T14:30:00
  │     │    │     │         │
  │     │    │     │         └─ Time slice
  │     │    │     └─────────── Depth in fractal
  │     │    └───────────────── Triangular path
  │     └────────────────────── Region
  └──────────────────────────── World
```

View them in the Context Panel when you retrieve memories.

## Troubleshooting

### API won't start
- Check if port 8001 is available: `lsof -i :8001`
- Install dependencies: `pip install fastapi uvicorn pydantic`

### Tests failing
- Make sure API is running first
- Check `data/fgms_memory.db` is writable

### Console not loading
- Check Node version: `node --version` (need 18+)
- Install dependencies: `cd web && npm install`
- Check port 3000: `lsof -i :3000`

## Architecture Overview

```
┌─────────────────┐
│  Memory Console │ ← http://localhost:3000/memory-console
└────────┬────────┘
         │
    HTTP/REST
         │
┌────────▼────────┐
│   FastAPI API   │ ← http://localhost:8001
│  • write        │
│  • read         │
│  • chat         │
│  • stats        │
└────────┬────────┘
         │
┌────────▼────────┐
│ SQLite Storage  │ ← data/fgms_memory.db
│ • Fractal index │
│ • FTS search    │
│ • Glyphs        │
└─────────────────┘
```

## Key Files

- `src/memory_system/` - Core Python implementation
- `web/app/memory-console/` - React UI
- `scripts/start_memory_api.py` - API launcher
- `scripts/test_memory_api.py` - Smoke tests
- `scripts/bench_*.py` - Benchmarks
- `data/fgms_memory.db` - SQLite database

## Documentation

- **Full guide**: `docs/PHASE-4-MEMORY-SYSTEM.md`
- **API docs**: http://localhost:8001/docs
- **PR description**: `docs/PHASE-4-PR-DESCRIPTION.md`

## What's Next?

- Read `docs/PHASE-4-MEMORY-SYSTEM.md` for deep dive
- Integrate with your own LLM
- Extend with custom policies
- Deploy to production
- Start Phase 5: Glyph-aware training

## Support

- Issues: https://github.com/funwae/fractal-glyph-tape/issues
- Email: contact@glyphd.com

---

**Enjoy exploring fractal memory! 🚀**
