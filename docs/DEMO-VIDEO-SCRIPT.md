# Fractal Glyph Memory — Demo Video Script

Approx length: 5–7 minutes
Audience: AI devs, infra people, smart enthusiasts

---

## 0. Cold Open (15–20 seconds)

**Screen:** Title card: *"Fractal Glyph Memory — A New Memory OS for AI Agents"*

**Voiceover:**

> "Everyone's talking about agents with memory, but most of them are just shoving JSON blobs and logs into a vector database. In this demo, I want to show you something different: a fractal, glyph-based memory system that compresses what an agent knows, makes it inspectable, and lets you pack more meaning into the same context window."

Cut to browser with your `/memory-console` open.

---

## 1. What Problem Are We Solving? (30–45 seconds)

**Screen:** Split: left = slides / text, right = small console preview.

**Voiceover:**

> "Modern LLM agents hit three walls really fast:
>
> 1. They forget things, because context windows are finite and expensive.
> 2. Their 'memories' are opaque — you can't see what they think is important.
> 3. Storage and retrieval get expensive as logs and history pile up.
>
> Fractal Glyph Memory attacks all three by treating memory as a **fractal tape** of repeated phrases and patterns, encoded into reusable glyphs that live in a structured address space."

Bullet list appears:

- Phrase families → glyphs
- Fractal addresses → multi-scale memory
- Foveated reads → efficient context under budget

---

## 2. Show the Agent in the Memory Console (1.5–2 minutes)

**Screen:** Full `/memory-console` view.

**Voiceover, while typing:**

> "This is the Fractal Glyph Memory Console. On the left is a normal chat with an agent. On the right, you can watch its memory in real time."

**Action 1 — establish actor & normal chat**

- Set `actor_id` to something like `hayden`.
- Send a few messages:

  1. "Hey, I'm working on a project called Fractal Glyph Tape. It compresses repeated phrases into glyphs."
  2. "I care a lot about context window efficiency and cheap long-term memory."
  3. "Also, remind me later that I prefer dark mode UIs and I'm in CST."

**Voiceover while you chat:**

> "I'm going to talk to this agent the way I normally would. I'll mention a project, some preferences, and a reminder."

**Action 2 — point at memory panel updating**

- After each send, briefly pause and move cursor to the right panel where the context, glyphs, and addresses update.

**Voiceover:**

> "Every time I send a message, the memory service writes it into a **fractal tape**:
>
> - It breaks my text into phrase families,
> - assigns each family a glyph — using Mandarin characters as pure symbols, not language,
> - and stores them at a precise address in a fractal layout: world, region, tri-path, depth, and time."

Zoom/hover:

- Highlight a glyph list entry: `谷阜`, `嶽岭`, etc.
- Highlight an address string like `earthcloud/hayden-agent#573@d2t17`.

---

## 3. Demonstrate Recall Under Token Budget (1–1.5 minutes)

**Action 3 — fast-forward time**

- Send a bunch of "later" messages that would normally push old context out of a window, e.g.:

  - talk about a different project,
  - some random questions,
  - anything that would realistically bury the original info.

**Voiceover:**

> "In a normal agent, those early details would eventually scroll off the end of the context window or get lost in a blob of embeddings. Here, they're compressed into glyphs and stored at addresses, so we can still reach them under a tight token budget."

**Action 4 — ask recall questions**

Now ask:

- "What did I say my current project is called and what does it do?"
- "What did I say I care about in terms of AI performance?"
- "What reminder did I leave you, and what time zone am I in?"

Let the agent respond.

**Voiceover:**

> "Behind the scenes, the agent isn't just stuffing the whole history in. It calls the **Fractal Glyph Memory Service** with a query and a token budget — say two thousand tokens."

**Action 5 — show `/memory/read` context**

- Click into the context panel for that last turn.
- Show that FGMS returned:
  - glyphs,
  - summaries/excerpts,
  - addresses,
  - and a token estimate.

**Voiceover:**

> "The memory service performs a **foveated read**:
>
> - It looks for the most relevant regions in the tape,
> - pulls shallow summaries from across your history,
> - and dives deep only where needed,
> - all while keeping the total under a budget.
>
> The context panel here is literally what went into the model."

---

## 4. Peek Behind the Fractal / Glyph Curtain (1–1.5 minutes)

**Screen:** Switch to your `/explore` page (Phase 1 visualizer) or any cluster view.

**Voiceover:**

> "Underneath, this isn't magic. It's a concrete semantic compression layer."

**Action 6 — show cluster / fractal map**

- Open a cluster corresponding to a glyph used in the last answer.
- Show phrase variants inside it.

**Voiceover:**

> "Each of these clusters is a **phrase family**. Those phrases might be:
>
> - 'I'm working on Fractal Glyph Tape',
> - 'This project compresses phrases into glyphs',
> - 'My current work is a fractal memory system.'
>
> The system learns that these live in the same region of meaning and gives that region a glyph."

If you have multilingual phrases in that cluster:

> "Because the embeddings are multilingual, the same glyph can anchor the English, Chinese, or other-language ways of saying the same thing. That becomes a cross-lingual memory anchor."

---

## 5. Hit the "Saves Money" & "Feels Different" Points (45–60 seconds)

**Screen:** Show a simple chart or table if you have one, or just the console.

**Voiceover:**

> "Why does this matter beyond being cool to look at?
>
> 1. **More context per dollar.** Instead of brute-forcing bigger context windows, we pack repeated patterns into glyphs and only expand them when needed.
> 2. **Inspectable memory.** You can literally see what the agent remembers and where it lives — down to addresses and clusters.
> 3. **Reusability.** Once a phrase family is learned, multiple agents and systems can share it by speaking the same glyph language.
>
> For companies, that means smaller storage footprints, cheaper long-horizon reasoning, and a debug surface that isn't just vectors and logs."

If you have any benchmark numbers from Phase 5:

> "In our early benchmarks, for the same token budget, the fractal memory context preserves more relevant information than a naive raw-history truncation, and that shows up in task accuracy."

---

## 6. Close With Where It Goes Next (20–30 seconds)

**Screen:** Back to console or a simple slide.

**Voiceover:**

> "What you've seen here is **Phase 4** of the system: a production-ready memory layer, with a glyph codec, foveated reads, SQLite-backed storage, and a live console.
>
> Phase 5 trains glyph-aware models that can speak this glyph language natively. Phase 6 extends the tape to video and other modalities, so agents can walk both text and visuals in the same fractal space.
>
> If you're building agents, copilots, or any AI that needs real memory, Fractal Glyph Memory is the substrate I wish existed years ago. Now it does."

**End card:**

- Repo link
- Your contact
- Short tagline, e.g.: *"Fractal Glyph Memory — a new substrate for AI memories."*
