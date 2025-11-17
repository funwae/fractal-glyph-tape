# Phase 6A — Agent Chat Integration with FGMS (FGT-FOVEATED)

**Goal**: Wire the `/api/agent/chat` endpoint to use the Fractal Glyph Memory Service with the same FGT-FOVEATED policy that achieved **+46.7pp accuracy improvement** in Phase 5 benchmarks.

This makes the live agent behavior match the measured benchmark story.

---

## 1. High-Level Flow

On each chat turn:

1. **User** sends a message to `/api/agent/chat`.
2. **Backend**:
   - Writes the user message into memory via internal write operation
   - Calls the foveation engine with:
     - `actor_id`
     - A query derived from the latest user message
     - A `token_budget` (e.g. 256/512/1024)
     - `mode="foveated"` (new mode for FGT-FOVEATED policy)
3. **FGMS** applies its **foveated policy** (matching Phase 5 FGT-FOVEATED):
   - ~30% budget to early turns
   - ~30% to semantically relevant turns
   - ~40% to the recent window
4. **Backend** builds a prompt:
   - `[System prompt] + [memory context from FGMS] + [recent messages]`
5. **Backend** calls the LLM provider (OpenAI, Claude, etc.)
6. **Backend**:
   - Writes the assistant reply to memory
   - Returns the reply (and optionally debug info) to the client

The **Memory Console UI** calls `/api/agent/chat` and displays the memory context used.

---

## 2. API Contract (Already Exists)

The `/api/agent/chat` endpoint already exists from Phase 4. We're enhancing it to use the new FGT-FOVEATED policy.

### 2.1 Request

Endpoint: `POST /api/agent/chat`

```json
{
  "actor_id": "hayden",
  "messages": [
    {"role": "user", "content": "Hey, what's the status of my Fractal Glyph Tape project?"},
    {"role": "assistant", "content": "Last time you said..."},
    {"role": "user", "content": "Yeah, and I also mentioned preferences about memory benchmarks."}
  ],
  "token_budget": 512,
  "mode": "foveated"  // "recent", "relevant", "mixed", or "foveated"
}
```

* `messages` follows OpenAI/ChatML structure (all previous turns the UI wants to send)
* `token_budget` is how many tokens we allow for **memory context**, not total prompt
* `mode` selects which foveation policy to use

### 2.2 Response

```json
{
  "response": "Here's what you said about your Fractal Glyph Tape project...",
  "memory_context": {
    "memories": [...],
    "addresses": [...],
    "glyphs": [...],
    "token_estimate": 248,
    "policy": "foveated",
    "candidates_considered": 45,
    "memories_selected": 8
  },
  "memories_used": 8,
  "tokens_used": 248
}
```

The UI can use `memory_context` to populate the right-hand panel in the Memory Console.

---

## 3. Implementation Plan

### 3.1 New Policy: `FoveatedPolicy`

Create a new foveation policy class in `src/memory_system/foveation/policies.py`:

**File**: `src/memory_system/foveation/policies.py`

**New Class**: `FoveatedPolicy`

**Algorithm**:

```python
class FoveatedPolicy(FoveationPolicy):
    """
    FGT-FOVEATED policy from Phase 5 benchmark.

    Allocates budget across three zones:
    - 30% to early turns (first 1-3 turns)
    - 30% to relevant turns (semantic match to query)
    - 40% to recent turns (last N turns)
    """

    def __init__(self, early_weight=0.30, relevant_weight=0.30, recent_weight=0.40):
        self.early_weight = early_weight
        self.relevant_weight = relevant_weight
        self.recent_weight = recent_weight

    def select_memories(
        self,
        memories: List[MemoryEntry],
        query: Optional[str],
        token_budget: int
    ) -> List[MemoryEntry]:
        """
        Select memories using three-zone allocation.

        1. Allocate sub-budgets to each zone
        2. Select memories for each zone independently
        3. Merge and deduplicate
        4. Return in chronological order
        """
        if not memories:
            return []

        # Sort memories by creation time
        sorted_by_time = sorted(memories, key=lambda m: m.created_at)

        # Zone budgets
        early_budget = int(token_budget * self.early_weight)
        relevant_budget = int(token_budget * self.relevant_weight)
        recent_budget = int(token_budget * self.recent_weight)

        # Zone 1: EARLY (first 1-3 turns)
        early_memories = self._select_early(sorted_by_time, early_budget)

        # Zone 2: RELEVANT (semantic match to query)
        relevant_memories = self._select_relevant(memories, query, relevant_budget)

        # Zone 3: RECENT (last N turns)
        recent_memories = self._select_recent(sorted_by_time, recent_budget)

        # Merge and deduplicate
        selected = self._merge_zones(early_memories, relevant_memories, recent_memories)

        # Respect total budget (in case of overlap)
        final = []
        tokens_used = 0
        for memory in sorted(selected, key=lambda m: m.created_at):
            if tokens_used + memory.token_estimate <= token_budget:
                final.append(memory)
                tokens_used += memory.token_estimate

        return final

    def _select_early(self, sorted_memories, budget):
        """Select first 1-3 turns within budget."""
        selected = []
        tokens_used = 0
        for memory in sorted_memories[:3]:  # First 3 turns max
            if tokens_used + memory.token_estimate <= budget:
                selected.append(memory)
                tokens_used += memory.token_estimate
        return selected

    def _select_relevant(self, memories, query, budget):
        """Select semantically relevant memories within budget."""
        if not query:
            return []

        # Score by keyword overlap (TODO: use embeddings in future)
        query_terms = set(query.lower().split())

        def relevance_score(memory):
            text_terms = set(memory.text.lower().split())
            intersection = query_terms & text_terms
            union = query_terms | text_terms
            return len(intersection) / len(union) if union else 0.0

        sorted_by_relevance = sorted(
            memories,
            key=relevance_score,
            reverse=True
        )

        selected = []
        tokens_used = 0
        for memory in sorted_by_relevance:
            if tokens_used + memory.token_estimate <= budget:
                selected.append(memory)
                tokens_used += memory.token_estimate

        return selected

    def _select_recent(self, sorted_memories, budget):
        """Select most recent memories within budget."""
        selected = []
        tokens_used = 0
        for memory in reversed(sorted_memories):
            if tokens_used + memory.token_estimate <= budget:
                selected.insert(0, memory)  # Insert at front to maintain order
                tokens_used += memory.token_estimate

        return selected

    def _merge_zones(self, early, relevant, recent):
        """Merge three zones, deduplicating by entry_id."""
        seen = set()
        merged = []

        for memory in early + relevant + recent:
            if memory.entry_id not in seen:
                seen.add(memory.entry_id)
                merged.append(memory)

        return merged
```

### 3.2 Update Foveation Engine

**File**: `src/memory_system/foveation/engine.py`

**Update**: Add `foveated` mode to the policy selector:

```python
def retrieve(self, actor_id, query, token_budget, mode="mixed", **kwargs):
    """Retrieve memories using specified policy."""

    # Select policy
    if mode == "recent":
        policy = RecentPolicy()
    elif mode == "relevant":
        policy = RelevantPolicy()
    elif mode == "mixed":
        policy = MixedPolicy()
    elif mode == "foveated":
        policy = FoveatedPolicy()  # NEW
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # ... rest of method
```

### 3.3 Update Agent Chat Endpoint

**File**: `src/memory_system/api/app.py`

**Update**: The endpoint already exists. We just need to ensure it supports the `foveated` mode, which it will once we add the policy.

**Optionally**: Update the default mode to `foveated` in the `AgentChatRequest` model:

```python
class AgentChatRequest(BaseModel):
    # ...
    mode: str = Field("foveated", description="Memory policy mode")  # Change default
```

---

## 4. Align with Phase 5 Benchmark

The `FoveatedPolicy` implementation above **exactly matches** the algorithm described in Phase 5:

- ✅ 30% budget to early turns (first 1-3)
- ✅ 30% budget to relevant turns (keyword/embedding match)
- ✅ 40% budget to recent turns (last N)
- ✅ Deduplication across zones
- ✅ Total budget enforcement

This ensures the **live agent behavior** is grounded in the **measured benchmark improvements**.

---

## 5. Memory Console Integration

**Current State**: The Memory Console already calls `/api/agent/chat` (from Phase 4).

**Update Needed**: Allow the user to select `foveated` mode in the UI.

**File**: `web/components/memory-console/ChatInterface.tsx`

**Change**: Add "Foveated" to the mode selector dropdown:

```tsx
<select value={mode} onChange={(e) => setMode(e.target.value)}>
  <option value="recent">Recent</option>
  <option value="relevant">Relevant</option>
  <option value="mixed">Mixed</option>
  <option value="foveated">Foveated (Phase 5)</option>  {/* NEW */}
</select>
```

The console will show exactly the memory the agent saw at each step, making it easy to debug and demonstrate.

---

## 6. Testing Plan

### 6.1 Unit Tests

**File**: `tests/test_foveated_policy.py`

Test cases:
- ✅ Early selection works (first 3 turns)
- ✅ Relevant selection works (keyword matching)
- ✅ Recent selection works (last N turns)
- ✅ Deduplication works
- ✅ Budget enforcement works
- ✅ Empty memory list returns empty

### 6.2 Integration Test

**File**: `tests/test_agent_chat_foveated.py`

Test scenario:
1. Create a synthetic episode (similar to Phase 5)
2. Write early setup memory
3. Write filler memories
4. Call `/api/agent/chat` with `mode=foveated` and `token_budget=256`
5. Verify early setup memory is included in response

### 6.3 Manual Testing

```bash
# 1. Start API
python scripts/start_memory_api.py --reload

# 2. Load synthetic episode
python scripts/load_test_episode.py --actor test_user --episode data/phase5_test_episode.json

# 3. Query with 256 token budget
curl -X POST http://localhost:8001/api/agent/chat \
  -H "Content-Type: application/json" \
  -d '{
    "actor_id": "test_user",
    "messages": [{"role": "user", "content": "What database did I say I prefer?"}],
    "token_budget": 256,
    "mode": "foveated"
  }'

# 4. Verify response includes early context
# Expected: Response should mention PostgreSQL (from early setup)
```

---

## 7. Success Criteria

Phase 6A is complete when:

- ✅ `FoveatedPolicy` class implemented in `policies.py`
- ✅ Foveation engine supports `mode=foveated`
- ✅ `/api/agent/chat` uses `foveated` mode
- ✅ Memory Console UI has "Foveated" option
- ✅ Unit tests pass for all policy methods
- ✅ Integration test passes for synthetic episode
- ✅ Manual test with 256-token budget retrieves early context

---

## 8. Future Enhancements

### 8.1 Embedding-Based Relevance

Replace keyword matching with semantic embeddings:

```python
def _select_relevant(self, memories, query, budget):
    # TODO: Use sentence-transformers to compute similarity
    query_embedding = self.encoder.encode(query)

    def semantic_similarity(memory):
        memory_embedding = self.encoder.encode(memory.text)
        return cosine_similarity(query_embedding, memory_embedding)

    # ... rest of method
```

This will improve relevance scoring beyond simple keyword overlap.

### 8.2 Real LLM Integration

Replace the mock LLM with real providers:

```python
if request.llm_provider == "openai":
    import openai
    response = openai.ChatCompletion.create(
        model=request.model or "gpt-4",
        messages=prompt_messages
    )
    response_text = response.choices[0].message.content

elif request.llm_provider == "anthropic":
    import anthropic
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=request.model or "claude-3-sonnet",
        messages=prompt_messages
    )
    response_text = response.content[0].text
```

### 8.3 Multi-Session Memory

Extend the policy to pull from previous sessions:

- Zone 1: Early from **current session**
- Zone 2: Relevant from **all sessions**
- Zone 3: Recent from **current session**
- Zone 4 (NEW): Summary from **previous sessions** (10% budget)

---

## 9. Deployment Considerations

### 9.1 API Compatibility

The `/api/agent/chat` endpoint is **backward compatible**:
- Old clients using `mode=mixed` still work
- New clients can use `mode=foveated` for improved performance

### 9.2 Performance

The `FoveatedPolicy` has similar complexity to `MixedPolicy`:
- O(N) sorting by time
- O(N) relevance scoring
- O(N) merging

No significant performance regression expected.

### 9.3 Monitoring

Add metrics to track:
- Average memories selected per zone (early/relevant/recent)
- Budget utilization per zone
- Deduplication rate (% of memories in multiple zones)

---

## 10. Documentation Updates

After Phase 6A is complete:

- ✅ Update `README.md` with "Foveated Mode" in features
- ✅ Update `QUICKSTART_PHASE4.md` with foveated examples
- ✅ Add API docs for `mode=foveated` in OpenAPI/Swagger
- ✅ Update Memory Console docs with foveated policy explanation

---

## Conclusion

Phase 6A bridges the gap between **benchmark results** and **production behavior**. By implementing the exact FGT-FOVEATED policy that achieved +46.7pp accuracy improvement in Phase 5, we ensure the live agent delivers the same measurable benefits.

This makes the Phase 5 story **real**—not just a research result, but a feature users can enable and experience.

---

**Next Step**: Implement the `FoveatedPolicy` class and wire it into the foveation engine.

**Estimated Time**: 2-3 hours
**Complexity**: Medium (well-specified algorithm, clear integration points)
**Risk**: Low (backward compatible, well-tested)

---

**Ready to implement!** 🚀
