# Phase 5 Context Efficiency Results

**Dataset:** data/phase5/dialog_test.jsonl
**Episodes:** 15
**Timestamp:** 2025-11-17T20:35:16.254668
**FGMS Available:** True

## Summary Table

| Token Budget | Strategy | Success Rate | Avg Tokens | Avg Completeness | Improvement |
|-------------|----------|--------------|------------|------------------|-------------|
| 256 | RAW-TRUNCATE | 26.7% (4/15) | 243 | 83.7% | baseline |
| 256 | FGT-FOVEATED | 73.3% (11/15) | 194 | 65.6% | +46.7pp |
| 512 | RAW-TRUNCATE | 73.3% (11/15) | 304 | 100.0% | baseline |
| 512 | FGT-FOVEATED | 73.3% (11/15) | 288 | 95.7% | +0.0pp |
| 1024 | RAW-TRUNCATE | 73.3% (11/15) | 304 | 100.0% | baseline |
| 1024 | FGT-FOVEATED | 73.3% (11/15) | 304 | 100.0% | +0.0pp |
| 2048 | RAW-TRUNCATE | 73.3% (11/15) | 304 | 100.0% | baseline |
| 2048 | FGT-FOVEATED | 73.3% (11/15) | 304 | 100.0% | +0.0pp |

## Key Findings

- **Average improvement:** +11.7 percentage points
- **Best improvement:** +46.7pp
- **Worst improvement:** +0.0pp

**FGT-CONTEXT shows advantages when:**
- Information needed for answering is mentioned early in conversation
- Token budget is limited relative to conversation length
- Relevant context is spread across non-contiguous turns

## Detailed Results by Budget

### Token Budget: 256

**Episodes tested:** 15

**RAW-TRUNCATE:**
- Correct answers: 4 / 15 (26.7%)
- Average tokens used: 243
- Average context completeness: 83.7%

**FGT-FOVEATED:**
- Correct answers: 11 / 15 (73.3%)
- Average tokens used: 194
- Average context completeness: 65.6%
- **Improvement: +46.7 percentage points**

### Token Budget: 512

**Episodes tested:** 15

**RAW-TRUNCATE:**
- Correct answers: 11 / 15 (73.3%)
- Average tokens used: 304
- Average context completeness: 100.0%

**FGT-FOVEATED:**
- Correct answers: 11 / 15 (73.3%)
- Average tokens used: 288
- Average context completeness: 95.7%
- **Improvement: +0.0 percentage points**

### Token Budget: 1024

**Episodes tested:** 15

**RAW-TRUNCATE:**
- Correct answers: 11 / 15 (73.3%)
- Average tokens used: 304
- Average context completeness: 100.0%

**FGT-FOVEATED:**
- Correct answers: 11 / 15 (73.3%)
- Average tokens used: 304
- Average context completeness: 100.0%
- **Improvement: +0.0 percentage points**

### Token Budget: 2048

**Episodes tested:** 15

**RAW-TRUNCATE:**
- Correct answers: 11 / 15 (73.3%)
- Average tokens used: 304
- Average context completeness: 100.0%

**FGT-FOVEATED:**
- Correct answers: 11 / 15 (73.3%)
- Average tokens used: 304
- Average context completeness: 100.0%
- **Improvement: +0.0 percentage points**
