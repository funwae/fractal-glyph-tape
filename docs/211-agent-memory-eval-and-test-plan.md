# Phase 4 Evaluation & Test Plan — Agent Memory OS

**Goal:** Prove that the Fractal Glyph Memory Service (FGMS) is:

- **Correct** — writes and reads do what we claim.
- **Efficient** — uses fewer tokens/bytes than naive baselines.
- **Useful** — improves downstream LLM behavior in realistic tasks.

---

## 1. Test Layers Overview

We test at four layers:

1. **Unit tests** — models, store, policy, service.
2. **Integration tests** — FGMS end-to-end on synthetic data.
3. **Micro-benchmarks** — compression & context efficiency on small corpora.
4. **Scenario-level experiments** — realistic agent use cases with LLMs.

---

## 2. Unit Tests

### 2.1 Implemented Tests

✅ `tests/memory/test_memory_models.py`:
- FractalAddress creation and validation
- Address serialization/deserialization
- MemorySpan and MemoryRecord validation

✅ `tests/memory/test_memory_store.py`:
- InMemoryStore operations
- SQLiteMemoryStore persistence
- Region and address listing
- Statistics computation

### 2.2 Running Tests

```bash
# Run all tests
pytest tests/memory/

# Run with coverage
pytest tests/memory/ --cov=src/memory --cov-report=html

# Run specific test file
pytest tests/memory/test_memory_models.py -v
```

---

## 3. Integration Tests

Script: `scripts/test_memory_integration.py` (TODO)

### 3.1 Test Scenarios

1. **Write & Read Flow**
   - Write multiple messages for an actor
   - Read with different token budgets
   - Verify context items returned

2. **Multi-Region Test**
   - Write to multiple regions
   - Verify region isolation
   - Test cross-region queries

3. **Depth Foveation**
   - Write at different depths
   - Verify foveation policy respects depth

---

## 4. Micro-Benchmarks

Script: `scripts/bench_memory_vs_baselines.py` (TODO)

### 4.1 Metrics to Track

- **Storage efficiency**: bytes on disk (raw vs FGT-coded)
- **Context efficiency**: information retained vs token budget
- **Retrieval latency**: time to retrieve context
- **Glyph density**: ratio of glyph tokens to total tokens

---

## 5. Agent Experiments

Script: `scripts/run_agent_memory_experiment.py` (TODO)

### 5.1 Use Cases

1. **Multi-turn conversation memory**
   - Agent remembers context across sessions
   - Retrieves relevant history for queries

2. **Task tracking**
   - Agent maintains task list in memory
   - Updates and recalls tasks efficiently

3. **Preference learning**
   - Agent learns user preferences
   - Applies them in future interactions

---

## 6. Success Criteria

- **Correctness**: All unit tests pass
- **Efficiency**: 30%+ reduction in context tokens vs naive truncation
- **Utility**: Improved task completion in agent experiments
- **Scalability**: Handle 10K+ records per actor

---

## 7. Current Status

✅ Unit tests implemented and passing
⏳ Integration tests (planned)
⏳ Micro-benchmarks (planned)
⏳ Agent experiments (planned)
