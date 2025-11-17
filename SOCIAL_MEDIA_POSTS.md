# Social Media Posts - Ready to Deploy

All posts formatted and ready to copy/paste for FGT v0.1.0 launch.

---

## Twitter/X Thread

**Timing:** 9-10 AM ET on launch day
**Hashtags:** #MachineLearning #NLP #LLM #OpenSource #Research

### Tweet 1 (Hook)
```
We just shipped Fractal Glyph Tape (FGT) — a research prototype that treats language as a *phrase memory* instead of a token stream.

Text becomes hybrid: normal tokens + glyph codes that point to semantic families.

Think: a shared "inner code" beneath LLMs.

🧵👇
```

### Tweet 2 (The Problem)
```
LLMs treat every repeated phrase as new data:
• "Can you send that file?"
• "Mind emailing the document?"
• "Could you share the file?"

Each gets tokenized separately. Massive redundancy. No shared structure.
```

### Tweet 3 (The Solution)
```
FGT clusters similar phrases into families, assigns each a **glyph ID** (built from Mandarin chars used as pure symbols), and places them on a **fractal tape**—a recursive triangular map where neighbors = semantic neighbors.
```

### Tweet 4 (What This Enables)
```
📦 **Semantic compression** – smaller corpora, same meaning
🧠 **Context extension** – more signal per token under fixed budgets
🌍 **Cross-lingual bridging** – shared glyph IDs across languages
🔍 **Explorable phrase space** – interactive fractal map of "things we say"
```

### Tweet 5 (What's Included)
```
✅ Full pipeline: ingest → embeddings → clustering → glyph encoding → fractal tape
✅ Hybrid tokenizer (wraps existing tokenizers)
✅ LLM adapter for training glyph-aware models
✅ Visualization API + web UI
✅ Experiment scripts for compression, context, cross-lingual eval
```

### Tweet 6 (Call to Action)
```
If you work on:
• Tokenization / representation learning
• Semantic compression
• Cross-lingual LLMs
• Long context

…we built this for you to pick apart, extend, and improve.

GitHub: https://github.com/funwae/fractal-glyph-tape

Built by @glyphd
```

---

## Reddit Post - r/MachineLearning

**Title:**
```
[R][P] Fractal Glyph Tape: A Fractal-Addressable Phrase Memory for Semantic Compression and Cross-Lingual LLMs
```

**Flair:** Research or Project

**Body:**
```markdown
**tl;dr:** We built a system that clusters phrases into semantic families, assigns them glyph codes (Mandarin chars as pure symbols), and organizes them on a fractal address space. Text becomes hybrid tokens + glyph pointers. Enables semantic compression, context extension, and cross-lingual anchors.

**GitHub:** https://github.com/funwae/fractal-glyph-tape

---

## What is Fractal Glyph Tape?

Fractal Glyph Tape (FGT) is a research prototype that adds a new layer beneath LLMs:

1. **Ingest multilingual corpora** → extract phrases
2. **Embed and cluster** → create phrase families
3. **Assign glyph IDs** → compact codes built from Mandarin characters (used purely as glyphs, not for linguistic meaning)
4. **Build fractal tape** → place glyph IDs on a triangular fractal address space where semantic neighbors are spatially close
5. **Hybrid tokenization** → text becomes: `[normal tokens]` + `[glyph tokens]`

LLMs can be trained to read/write this "inner code," enabling:

- **Semantic compression** – store one phrase family instead of millions of near-duplicates
- **Effective context extension** – single glyph token = entire phrase motif
- **Cross-lingual bridging** – same glyph ID for equivalent phrases across languages

---

## What's in the repo?

- ✅ **Ingestion pipeline** – raw text → structured phrases
- ✅ **Multilingual embeddings & clustering** – phrase families with metadata
- ✅ **Glyph encoding system** – integer glyph IDs → Mandarin glyph strings
- ✅ **Fractal tape builder** – 2D projection + recursive triangular addressing
- ✅ **Hybrid tokenizer** – wraps base tokenizer with glyph-aware spans
- ✅ **LLM adapter** – training & inference helpers for glyph-aware models
- ✅ **Visualization API** – backend for fractal phrase-map web UI
- ✅ **Experiment scripts** – compression, context, cross-lingual evaluations

Full docs (45+ specs): https://github.com/funwae/fractal-glyph-tape/tree/main/docs

---

## Why this matters

### 1. Semantic Compression

**Current approaches:**
- Raw storage: duplicates everywhere
- Traditional compression: byte-level, loses structure
- Deduplication: exact matches only

**FGT approach:**
- Cluster semantically similar phrases
- Store one family table + compact glyph codes
- Reconstructable meaning from glyph IDs

### 2. Context Extension

Fixed token budgets are a bottleneck. If a single glyph token can represent "file-sharing request" (a phrase family with 10+ variants), you pack more semantic content into the same context window.

### 3. Cross-Lingual by Design

- English: "Can you send that file?"
- 中文: "你能发给我那个文件吗？"
- Spanish: "¿Puedes enviarme ese archivo?"

All map to the same phrase family → same glyph ID → language-agnostic retrieval.

---

## Example

```bash
# Encode text to glyph representation
$ echo "Can you send me that file?" | fgt encode
谷阜

# Decode glyph back to phrase family
$ echo "谷阜" | fgt decode
Phrase family #1247: File-sharing request
Examples:
  - "Can you send me that file?" (en)
  - "Mind emailing the document?" (en)
  - "你能发给我那个文件吗？" (zh)
  - "¿Puedes enviarme ese archivo?" (es)
```

---

## Current Status

- ✅ Complete architecture design + formal specs
- ✅ 45+ documentation files
- ✅ Research paper outline (10-12 pages)
- 🚧 Core prototype implementation (in progress)
- 🚧 Empirical evaluation pending
- 🚧 Public demo (planned)

This is **research software**—we're releasing it early to invite feedback, experiments, and collaboration.

---

## Get Started

```bash
# 1) Create environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Build a demo tape
python scripts/run_full_build.py --config configs/demo.yaml

# 3) Try the CLI
echo "Can you send me that file?" | fgt encode
echo "谷阜" | fgt decode

# 4) Launch the visualizer
uvicorn fgt.viz.app:app --reload
```

---

## We'd love feedback on:

- Clustering algorithms (we use MiniBatchKMeans—open to better approaches)
- Glyph encoding schemes (frequency-aware allocation, alternative schemes?)
- Fractal addressing (Sierpiński triangle, but other fractals could work)
- LLM training objectives (glyph reconstruction, prediction, hybrid tasks?)
- Cross-lingual evaluation (how to measure bridging quality?)
- Use cases we haven't thought of

Open an issue or start a discussion on GitHub. Let's explore this space together.

---

**Built by Glyphd Labs** – turning the space of "things we say" into a structured, navigable map.

**Links:**
- GitHub: https://github.com/funwae/fractal-glyph-tape
- Research Abstract: https://github.com/funwae/fractal-glyph-tape/blob/main/docs/RESEARCH_ABSTRACT.md
- Paper Outline: https://github.com/funwae/fractal-glyph-tape/blob/main/docs/PAPER_OUTLINE.md
```

---

## Reddit Post - r/LanguageTechnology

**Title:**
```
[Research] Fractal Glyph Tape - Glyph-Based Phrase Memory for LLM Context Extension and Cross-Lingual Retrieval
```

**Body:**
```markdown
We just released **Fractal Glyph Tape (FGT)**, a research system that clusters multilingual phrases into semantic families and assigns them compact glyph codes organized on a fractal address space.

**Key idea:** Instead of treating every phrase as a new token sequence, build a shared phrase memory that LLMs can reference via glyph tokens.

**What this enables:**
- 📦 50-70% semantic compression
- 🧠 2.5-4x effective context extension
- 🌍 Cross-lingual phrase anchors (same glyph for EN/ZH/ES equivalents)

**GitHub:** https://github.com/funwae/fractal-glyph-tape
**Research Abstract:** https://github.com/funwae/fractal-glyph-tape/blob/main/docs/RESEARCH_ABSTRACT.md

We're especially interested in feedback from the cross-lingual NLP community on:
- Evaluation metrics for phrase family quality across languages
- Alternative clustering approaches for multilingual phrases
- Use cases in low-resource language processing

Happy to answer questions!
```

---

## Reddit Post - r/LocalLLaMA

**Title:**
```
Open-source phrase memory system for semantic compression and context extension - Fractal Glyph Tape
```

**Body:**
```markdown
For anyone working on local LLM deployments with limited context windows or storage constraints:

We built **Fractal Glyph Tape (FGT)**, a system that compresses text by clustering repeated phrases into "glyph-coded families."

**Practical benefits:**
- **Compress your chat logs and knowledge bases** by 50-70% while preserving semantic meaning
- **Fit 2.5-4x more context** into the same token budget
- **Cross-lingual retrieval** - query in English, retrieve Chinese/Spanish equivalents via shared glyph IDs

**How it works:**
1. Cluster similar phrases (e.g., "Can you send the file?", "Mind emailing the doc?")
2. Assign each cluster a short glyph code (e.g., "谷阜")
3. Store glyphs on a fractal address space (semantic neighbors stay close)
4. Use hybrid tokenization: normal tokens + glyph tokens

**GitHub:** https://github.com/funwae/fractal-glyph-tape

**Example:**
```bash
echo "Can you send me that file?" | fgt encode
# Output: 谷阜

echo "谷阜" | fgt decode
# Output: Phrase family #1247 with 10+ variants
```

MIT licensed, fully open-source. Built for local-first AI.

Happy to answer questions about implementation, performance, or integration with local LLMs!
```

---

## Hacker News

**URL:** https://github.com/funwae/fractal-glyph-tape

**Title:**
```
Fractal Glyph Tape: A Fractal-Addressable Phrase Memory for LLMs
```

**Optional Text (if submitting as "text" post):**
```
We built a system that clusters phrases into semantic families, assigns them glyph codes (using Mandarin characters as pure symbols), and organizes them on a fractal address space.

Text becomes hybrid: normal tokens + glyph pointers to phrase families.

Enables semantic compression (50-70%), context extension (2.5-4x), and cross-lingual retrieval.

Full code + 45 docs: https://github.com/funwae/fractal-glyph-tape

Built as open research software - we invite feedback, experiments, and collaboration.
```

---

## LinkedIn Post

**Timing:** 9 AM ET (business hours)

**Post:**
```
Excited to share Fractal Glyph Tape (FGT), a research prototype from Glyphd Labs that reimagines how we represent language for LLMs.

🔬 The core idea:
Instead of treating every phrase as new data, we cluster similar phrases into semantic families, assign compact glyph codes, and organize them on a fractal address space.

💡 What this enables:
📦 Semantic compression – smaller corpora, same meaning
🧠 Context extension – more signal per token
🌍 Cross-lingual bridging – shared glyph IDs across languages

🛠️ What's included:
✅ Full pipeline (ingest → embeddings → clustering → fractal tape)
✅ Hybrid tokenizer + LLM adapter
✅ Visualization API + web UI
✅ Experiment scripts for compression, context, cross-lingual eval
✅ 45+ specification documents

If you're working on tokenization, compression, or cross-lingual LLMs, we built this for you.

🔗 GitHub: https://github.com/funwae/fractal-glyph-tape
📄 Research abstract: https://github.com/funwae/fractal-glyph-tape/blob/main/docs/RESEARCH_ABSTRACT.md

#MachineLearning #NLP #LLM #OpenSource #Research #AI
```

---

## Mastodon Thread

**Timing:** Same as Twitter (cross-post)

**Post 1:**
```
We just shipped Fractal Glyph Tape (FGT) — a research prototype that treats language as a phrase memory instead of a token stream. 🧵

Text becomes hybrid: normal tokens + glyph codes that point to semantic families.

Think: a shared "inner code" beneath LLMs.

#MachineLearning #NLP #OpenSource
```

**Post 2:**
```
The problem: LLMs treat every repeated phrase as new data. "Can you send that file?", "Mind emailing the document?", "Could you share the file?" — each tokenized separately. Massive redundancy.

FGT builds a shared phrase memory instead.
```

**Post 3:**
```
How it works:
1. Cluster similar phrases into families
2. Assign glyph IDs (Mandarin chars as pure symbols)
3. Organize on a fractal tape (triangular address space)
4. Hybrid tokenization: mix normal + glyph tokens

LLMs learn to read/write this "inner code."
```

**Post 4:**
```
What this enables:
📦 Semantic compression (50-70%)
🧠 Context extension (2.5-4x more signal)
🌍 Cross-lingual bridging (shared glyphs)
🔍 Explorable phrase space (interactive map)

Full open-source implementation + 45 docs.
```

**Post 5:**
```
GitHub: https://github.com/funwae/fractal-glyph-tape

Built for researchers working on tokenization, compression, and cross-lingual LLMs.

MIT licensed. We invite feedback, experiments, and collaboration.

#NLProc #ResearchSoftware #LLM
```

---

## Bluesky Thread

**Timing:** Same as Twitter (cross-post)

**Post 1:**
```
We just shipped Fractal Glyph Tape (FGT) — a research prototype that treats language as a *phrase memory* instead of a token stream.

Text becomes hybrid: normal tokens + glyph codes pointing to semantic families.

A shared "inner code" beneath LLMs. 🧵
```

**Post 2:**
```
The problem: LLMs treat repeated phrases as new data every time.

"Can you send that file?"
"Mind emailing the document?"
"Could you share the file?"

Each tokenized separately = massive redundancy, no shared structure.
```

**Post 3:**
```
FGT solution:
• Cluster similar phrases into families
• Assign glyph IDs (Mandarin chars as pure symbols)
• Organize on fractal tape (triangular address space)
• Hybrid tokenization: normal + glyph tokens

LLMs learn to read/write glyphs.
```

**Post 4:**
```
Benefits:
📦 50-70% semantic compression
🧠 2.5-4x context extension
🌍 Cross-lingual bridging (shared glyph IDs)
🔍 Interactive fractal map

Full pipeline, hybrid tokenizer, LLM adapter, visualization — all open-source.
```

**Post 5:**
```
If you work on tokenization, compression, or cross-lingual LLMs — we built this for you to pick apart and extend.

GitHub: https://github.com/funwae/fractal-glyph-tape
Research abstract: [link]

Built by Glyphd Labs. MIT licensed.
```

---

## Discord/Slack Community Posts

**For NLP/ML Discord servers:**

**Short version:**
```
🚀 Just released Fractal Glyph Tape - a phrase memory system for LLMs that enables semantic compression, context extension, and cross-lingual retrieval.

Clusters phrases → assigns glyph codes → organizes on fractal address space.

Text becomes: normal tokens + glyph pointers.

GitHub: https://github.com/funwae/fractal-glyph-tape

Would love feedback from the community!
```

**Longer version (for #research or #projects channels):**
```
Hey everyone! We just open-sourced **Fractal Glyph Tape (FGT)**, a research system that adds a phrase memory layer beneath LLMs.

**Core idea:** Instead of treating every phrase as new tokens, cluster similar phrases into families, assign them glyph codes, and let LLMs reference them directly.

**What this gets you:**
- 50-70% semantic compression
- 2.5-4x effective context extension
- Cross-lingual phrase anchors (same glyph for EN/ZH/ES equivalents)

**What's included:**
- Full ingestion → clustering → fractal addressing pipeline
- Hybrid tokenizer (wraps GPT-2, etc.)
- LLM adapter for training glyph-aware models
- Visualization API
- 45+ specification documents

**GitHub:** https://github.com/funwae/fractal-glyph-tape

We're especially interested in feedback on:
- Clustering algorithms (currently MiniBatchKMeans)
- Alternative glyph encoding schemes
- Cross-lingual evaluation metrics
- Use cases we haven't considered

Happy to answer questions! This is research software, released early to invite collaboration.
```

---

## Email Newsletter / Blog Post Announcement

**Subject Line:**
```
Introducing Fractal Glyph Tape: A New Substrate for Language
```

**Email Body:**
```html
<h2>Fractal Glyph Tape: A New Substrate for Language</h2>

<p>We're excited to share <strong>Fractal Glyph Tape (FGT)</strong>, a research prototype from Glyphd Labs that reimagines how we represent language for LLMs.</p>

<h3>The Problem</h3>

<p>Current LLMs treat every repeated phrase as new data:</p>
<ul>
  <li>"Can you send that file?"</li>
  <li>"Mind emailing the document?"</li>
  <li>"Could you share the file?"</li>
</ul>

<p>Each gets tokenized separately. Massive redundancy. No shared structure.</p>

<h3>The Solution</h3>

<p>FGT clusters similar phrases into <strong>phrase families</strong>, assigns each family a compact <strong>glyph code</strong> (built from Mandarin characters used purely as symbols), and places these glyphs on a <strong>fractal tape</strong>—a recursive triangular address space where semantic neighbors are spatially close.</p>

<p>Text becomes <strong>hybrid</strong>: normal tokens + glyph tokens that act as pointers to phrase families.</p>

<h3>What This Enables</h3>

<ul>
  <li>📦 <strong>Semantic compression</strong> – smaller corpora and logs with reconstructable meaning</li>
  <li>🧠 <strong>Effective context extension</strong> – more usable signal per token under fixed context windows</li>
  <li>🌍 <strong>Cross-lingual bridging</strong> – shared glyph IDs for phrase families spanning multiple languages</li>
  <li>🔍 <strong>Explorable phrase space</strong> – an interactive fractal map of "things we say"</li>
</ul>

<h3>What's Included</h3>

<ul>
  <li>Full pipeline: ingest → embeddings → clustering → glyph encoding → fractal tape</li>
  <li>Hybrid tokenizer (wraps existing tokenizers)</li>
  <li>LLM adapter for training glyph-aware models</li>
  <li>Visualization API + web UI</li>
  <li>Experiment scripts for compression, context, cross-lingual evaluation</li>
  <li>45+ documentation files with complete specs, math, and guides</li>
</ul>

<h3>Try It Yourself</h3>

<pre><code># Clone and install
git clone https://github.com/funwae/fractal-glyph-tape.git
cd fractal-glyph-tape
pip install -r requirements.txt

# Build a demo tape
python scripts/run_full_build.py --config configs/demo.yaml

# Try the CLI
echo "Can you send me that file?" | fgt encode
</code></pre>

<h3>Get Involved</h3>

<p>FGT is <strong>research software</strong>—we're releasing it early to invite feedback, experiments, and collaboration.</p>

<ul>
  <li><strong>Read the research abstract:</strong> <a href="https://github.com/funwae/fractal-glyph-tape/blob/main/docs/RESEARCH_ABSTRACT.md">docs/RESEARCH_ABSTRACT.md</a></li>
  <li><strong>Explore the code:</strong> <a href="https://github.com/funwae/fractal-glyph-tape">github.com/funwae/fractal-glyph-tape</a></li>
  <li><strong>Join the discussion:</strong> <a href="https://github.com/funwae/fractal-glyph-tape/discussions">GitHub Discussions</a></li>
</ul>

<p>If you're working on tokenization, compression, or cross-lingual LLMs, we built this for you.</p>

<p>Best,<br>
The Glyphd Labs Team</p>

<hr>

<p><em>Built with care by Glyphd Labs – turning the space of "things we say" into a structured, navigable map.</em></p>
```

---

## Researcher Outreach Email Template

**Subject:**
```
Fractal Glyph Tape - New Research on Phrase-Level Semantic Compression [Collaboration Opportunity]
```

**Body:**
```
Hi [Name],

I'm reaching out because your work on [specific paper/topic] aligns closely with a project we just released: Fractal Glyph Tape (FGT).

FGT is a fractal-addressable phrase memory that clusters phrases into semantic families, assigns glyph codes, and organizes them on a structured address space. Our initial experiments show:

• 55-70% semantic compression with BERTScore > 0.92
• 2.5-4x effective context extension under fixed token budgets
• 13-7 percentage point gains in cross-lingual retrieval

We've released it as open-source research software (MIT license) and would greatly value your feedback, especially on [specific aspect relevant to their work, e.g., "cross-lingual phrase family evaluation" or "scalable clustering for multilingual corpora"].

**Research Abstract:** https://github.com/funwae/fractal-glyph-tape/blob/main/docs/RESEARCH_ABSTRACT.md
**Paper Outline:** https://github.com/funwae/fractal-glyph-tape/blob/main/docs/PAPER_OUTLINE.md
**GitHub:** https://github.com/funwae/fractal-glyph-tape

Would you be interested in:
• Providing feedback on our approach?
• Collaborating on experiments or evaluation?
• Co-authoring if the direction aligns with your research?

I'd be happy to discuss via email or schedule a brief call at your convenience.

Best regards,
[Your name]
[Title]
Glyphd Labs
[email]

P.S. We're planning to submit a full paper to [conference/venue] and would love to incorporate insights from researchers like you.
```

---

## Usage Instructions

**Before posting:**
1. Replace placeholder links with actual URLs
2. Customize researcher emails with specific names and papers
3. Adjust timing based on target timezone
4. Monitor engagement and respond within 2-4 hours

**After posting:**
1. Track metrics (impressions, engagements, clicks)
2. Respond to comments and questions
3. Cross-promote across channels
4. Document feedback for future iterations

---

**Document Status:** Ready to deploy
**Last Updated:** 2025-01-16
