# Use Case: Agent Memory OS Powered by Fractal Glyph Memory

This document describes a concrete vertical: **agent memory**.

Goal:

> Provide a drop-in memory backend for AI assistants/agents that:
> - compresses and structures their histories,
> - enables long-horizon recall under token budgets,
> - exposes memory state visually and metrically.

## 1. Problem

Current agent memory patterns:

- Store raw messages / JSON blobs in a DB or vector store.
- At inference:
  - retrieve top-K chunks,
  - stuff them into the prompt until the context window bursts.

Problems:

- Redundant, unstructured histories.
- No multi-scale / z-depth notion of memory.
- Hard to introspect; invisible "blob memory."

## 2. FGM-based solution

For each actor/agent:

1. All interactions are streamed into FGMS via `/memory/write`.
2. FGT converts text to phrase families and glyph-coded entries.
3. Entries are addressed in fractal space with z-depth and time slices.
4. An "agent memory policy" uses `/memory/read` to retrieve context:
   - coarse summaries across regions,
   - deep dives where relevant.

Additionally:

- glyph-core / glyph-drive-3d render the memory state (coil + 3D drive).
- TGO tracks memory usage over time.

## 3. Typical integration flow

### 3.1 On each user/agent turn

1. User → Agent message:
   - Agent logs conversation step via `/memory/write`.
2. Agent decides next action:
   - Before generating, agent calls `/memory/read` with:
     - `actor_id`,
     - `query` constructed from current goal,
     - `token_budget` aligned with LLM context limit.
3. FGMS returns foveated context:
   - The agent sends it along with the latest messages into the LLM.

### 3.2 Example pseudo-code (TypeScript-ish)

```ts
const MEMORY_BUDGET = 2000;

async function agentStep(actorId, userMessage) {
  // 1) Write new message into memory
  await post("/api/memory/write", {
    actor_id: actorId,
    text: userMessage,
    tags: ["chat"]
  });

  // 2) Build a memory query
  const query = `What does ${actorId} care about related to: ${userMessage}?`;

  // 3) Get foveated context
  const memory = await post("/api/memory/read", {
    actor_id: actorId,
    query,
    token_budget: MEMORY_BUDGET,
    mode: "mixed"
  });

  // 4) Compose LLM prompt
  const prompt = buildPrompt(userMessage, memory.context);

  // 5) Call LLM
  const reply = await callLLM(prompt);

  // 6) Write LLM reply back into memory
  await post("/api/memory/write", {
    actor_id: actorId,
    text: reply,
    tags: ["assistant"]
  });

  return reply;
}
```

## 4. Metrics to track

Using TGO, we track:

* Memory size over time:

  * number of addresses per region,
  * bytes on disk.
* Glyph density:

  * fraction of content stored as glyph vs raw.
* Context efficiency:

  * average `token_estimate` for `/memory/read`,
  * performance on retrieval tasks.
* Drift:

  * how cluster assignments and addresses evolve across versions.

## 5. Demo plan

1. Collect a week/month of real chat / notes for a single user (your own).
2. Build FGT tape and run FGMS over it.
3. Implement a small chat UI that:

   * uses FGMS for memory,
   * shows glyph-core / glyph-drive visualizations alongside.
4. Instrument with TGO to show:

   * how memory grows,
   * how context budgets are used.

This becomes the **flagship demo** that makes Fractal Glyph Memory legible.
