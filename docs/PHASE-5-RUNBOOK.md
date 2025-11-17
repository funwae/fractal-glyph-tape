# Phase 5 Runbook — Glyph-Aware Training & Evaluation

**Status:** Phase 4 complete, Phase 5 scaffolding ready.
**Goal:** Produce hard evidence that glyph-aware context has advantages under fixed token budgets.

We focus on:

1. Running baseline vs glyph-aware experiments end-to-end.
2. Producing metrics + plots in `reports/phase5/`.
3. Writing a concise `PHASE-5-RESULTS.md` summarizing findings.

---

## 1. Datasets

### 1.1 Source

Use **two datasets**:

1. **Synthetic or personal agent logs** (internally generated or anonymized).
2. **A public multi-turn dialog dataset** (e.g. MultiWOZ-style or open QA/chat).

Store under:

- `data/phase5/agent_logs.jsonl`
- `data/phase5/public_dialog.jsonl`

Each record format:

```json
{
  "episode_id": "string",
  "turns": [
    {"role": "user", "text": "..."},
    {"role": "assistant", "text": "..."}
  ]
}
```

Claude task:

* Implement a script `scripts/phase5_prepare_datasets.py` that:

  * reads raw data,
  * normalizes to this format,
  * splits into `train/val/test`.

---

## 2. Context Efficiency Benchmark (No Training Yet)

Before full training, run **pure-context experiments** using the existing LLM adapter.

### 2.1 Script: `scripts/phase5_bench_context.py`

Implement a CLI:

```bash
python scripts/phase5_bench_context.py \
  --dataset data/phase5/agent_logs.jsonl \
  --budget 2048 \
  --out reports/phase5/context_bench_agent.json
```

Features:

* Sample N episodes (configurable, e.g. 500).

* For each episode:

  * Define a **question** about something that happened early in the episode.

    * Example: "What did the user decide about X earlier?" or "What constraint was set at the beginning?"
  * Run 2 strategies:

    1. **RAW-TRUNCATE**

       * Pass last N tokens of raw history to the LLM.
    2. **FGT-CONTEXT**

       * Use FGMS:

         * write whole episode into memory (if not already),
         * call `/memory/read` with `token_budget`,
         * construct prompt using returned glyph/text context.

* Use a judging model (config in code) to answer:

  * "Which answer is more correct w.r.t the ground-truth episode?"

Metrics:

* success rate for RAW-TRUNCATE vs FGT-CONTEXT.
* average `token_estimate` from FGMS.
* stored in JSON with schema:

```json
{
  "config": {...},
  "results": {
    "raw_truncate": {"success_rate": 0.XX, "avg_tokens": N},
    "fgt_context": {"success_rate": 0.XX, "avg_tokens": N}
  }
}
```

Output to `reports/phase5/context_bench_*.json`.

---

## 3. Glyph-Aware Mini-Model (Optional but Strong)

If compute allows, train a **tiny glyph-aware model** to test deeper integration.

### 3.1 Data prep

Script: `scripts/phase5_prepare_glyph_aware_dataset.py`

* Use FGMS to generate glyph-aware episodes:

  * For each episode in train/val/test:

    * Write all turns through `/memory/write`.
    * For each QA/summarization training example:

      * Get glyph-coded context via `/memory/read`.
      * Build `(input_prompt, target_output)` pairs for:

        * RAW baseline.
        * GLYPH-CONTEXT variant.

Store under:

* `data/phase5/glyph_aware/train.jsonl`
* `data/phase5/glyph_aware/val.jsonl`
* `data/phase5/glyph_aware/test.jsonl`

### 3.2 Training

Script: `scripts/phase5_train_glyph_model.py`

* Start from a small open model (configurable).

* Train two versions:

  1. `model_raw`: trained on RAW prompts.
  2. `model_glyph`: trained on glyph-context prompts.

* Use simple HF Trainer or custom loop:

  * moderate training steps (this is a proof-of-concept, not SOTA).

### 3.3 Evaluation

Script: `scripts/phase5_eval_glyph_model.py`

* Evaluate both models on:

  * context QA,
  * long-horizon summarization.
* Under fixed token budgets.
* Same JSON reporting format as context benchmarks, plus model name.

Save results under `reports/phase5/model_eval_*.json`.

---

## 4. Reporting & Plots

Script: `scripts/phase5_summarize_results.py`

* Aggregate all `reports/phase5/*.json`.
* Produce:

  * `reports/phase5/summary_table.md` — markdown table with key metrics.
  * `reports/phase5/plots/` — simple PNG charts (tokens vs success rate, etc).

Key questions to answer:

1. Does FGT-CONTEXT beat RAW-TRUNCATE under the same token budget?
2. How does glyph-context compare to raw for the same model size?
3. What are the regimes where FGT helps the most? (longer episodes, more redundancy, etc.)

---

## 5. Phase 5 Completion Criteria

We consider Phase 5 complete when:

1. `scripts/phase5_bench_context.py` runs successfully on at least 2 datasets.
2. `reports/phase5/summary_table.md` exists and clearly shows:

   * at least one setting where FGT-CONTEXT outperforms RAW-TRUNCATE.
3. (Optional) Glyph-aware model evaluation shows:

   * at least comparable performance with more compressed contexts.
4. `docs/PHASE-5-RESULTS.md` summarizes:

   * setup,
   * key numbers,
   * caveats,
   * and recommended claims to make publicly.

Claude's job: **implement and run everything in this runbook**, then draft `PHASE-5-RESULTS.md`.
