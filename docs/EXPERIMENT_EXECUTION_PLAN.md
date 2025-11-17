# Experiment Execution Plan

**Purpose:** Detailed protocols for running all experiments from `PAPER_OUTLINE.md` §5

**Timeline:** 2-4 weeks to complete all experiments

**Prerequisites:**
- FGT core implementation complete
- Datasets prepared and preprocessed
- Evaluation metrics implemented
- Compute resources available (GPU recommended)

---

## Overview

Five main experiments + ablation studies:

1. **Experiment 1:** Semantic Compression
2. **Experiment 2:** Context Window Extension
3. **Experiment 3:** Cross-Lingual Retrieval
4. **Experiment 4:** Cluster Quality
5. **Experiment 5:** Fractal Address Space Quality
6. **Ablations:** Glyph encoding schemes, fractal depth, clustering algorithms

---

## Experiment 1: Semantic Compression

### Goal
Demonstrate that FGT achieves semantic compression while preserving meaning.

### Hypothesis
FGT achieves 55-70% compression ratio with BERTScore > 0.92.

### Setup

**Dataset:**
- 100,000 phrases from OpenSubtitles (conversational, high repetition)
- 100,000 phrases from Wikipedia (encyclopedic, lower repetition)
- 100,000 phrases from customer support logs (domain-specific, high repetition)

**Baselines:**
1. **gzip** - byte-level compression
2. **BPE-only** - standard tokenization (no compression)
3. **Exact deduplication** - remove exact duplicates only
4. **FGT (ours)** - hybrid tokenization with glyph encoding

**Metrics:**
1. **Compression ratio:** `original_bytes / compressed_bytes`
2. **Semantic preservation:** BERTScore between original and reconstructed text
3. **Glyph coverage:** % of tokens that are glyphs vs. raw

### Protocol

**Step 1: Build phrase memory**
```bash
# Build FGT tape from training data
python scripts/build_tape.py \
  --input data/opensubtitles_100k.jsonl \
  --output models/tape_opensubs.db \
  --num-clusters 10000 \
  --embedding-model paraphrase-multilingual-mpnet-base-v2
```

**Step 2: Encode test set**
```bash
# Encode test phrases using hybrid tokenizer
python scripts/encode_corpus.py \
  --tape models/tape_opensubs.db \
  --input data/opensubtitles_test.jsonl \
  --output results/exp1_opensubs_encoded.jsonl \
  --base-tokenizer gpt2
```

**Step 3: Measure compression**
```python
import json
import os

def measure_compression(original_file, encoded_file):
    # Original size
    original_size = os.path.getsize(original_file)

    # Encoded size
    encoded_size = os.path.getsize(encoded_file)

    # Compression ratio
    ratio = original_size / encoded_size

    # Glyph coverage
    with open(encoded_file) as f:
        data = [json.loads(line) for line in f]
    total_tokens = sum(item['total_tokens'] for item in data)
    glyph_tokens = sum(item['glyph_tokens'] for item in data)
    coverage = glyph_tokens / total_tokens

    return {
        'compression_ratio': ratio,
        'glyph_coverage': coverage,
        'original_bytes': original_size,
        'encoded_bytes': encoded_size
    }

results = measure_compression(
    'data/opensubtitles_test.jsonl',
    'results/exp1_opensubs_encoded.jsonl'
)
print(json.dumps(results, indent=2))
```

**Step 4: Measure semantic preservation**
```python
from sentence_transformers import SentenceTransformer
from bert_score import score

model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# Load original and reconstructed texts
with open('data/opensubtitles_test.jsonl') as f:
    originals = [json.loads(line)['text'] for line in f]

with open('results/exp1_opensubs_reconstructed.jsonl') as f:
    reconstructed = [json.loads(line)['text'] for line in f]

# Compute BERTScore
P, R, F1 = score(reconstructed, originals, model_type='bert-base-multilingual-cased')

print(f"BERTScore F1: {F1.mean():.4f} (±{F1.std():.4f})")
```

**Step 5: Run baselines**
```bash
# gzip compression
gzip -c data/opensubtitles_test.jsonl > results/exp1_gzip.jsonl.gz
gzip -l results/exp1_gzip.jsonl.gz

# Exact deduplication
python scripts/deduplicate.py \
  --input data/opensubtitles_test.jsonl \
  --output results/exp1_dedup.jsonl
```

### Expected Results

| Method | Compression Ratio | BERTScore F1 | Glyph Coverage |
|--------|-------------------|--------------|----------------|
| FGT (conversational) | 0.55-0.70 | > 0.92 | 50-60% |
| FGT (encyclopedic) | 0.40-0.55 | > 0.90 | 30-40% |
| gzip | 0.70-0.80 | N/A (lossy) | N/A |
| BPE-only | 1.00 (no compression) | 1.00 | 0% |
| Dedup | 0.90-0.95 | 1.00 (exact) | N/A |

### Analysis

**Key findings to report:**
- Compression highest in conversational data (high repetition)
- Semantic preservation strong (BERTScore > 0.92)
- Trade-off: compression vs. coverage
- Comparison: FGT outperforms dedup, preserves semantics unlike gzip

### Deliverables

- [ ] Compression ratio table (Table 2 in paper)
- [ ] BERTScore distributions (figure)
- [ ] Glyph coverage analysis
- [ ] Per-domain breakdown

---

## Experiment 2: Context Window Extension

### Goal
Show that hybrid tokenization fits more semantic content in fixed token budgets.

### Hypothesis
Hybrid tokenization fits 2.5-4x more source text in same token budget, improving downstream task performance.

### Setup

**Tasks:**
1. **Summarization:** XSum dataset
2. **Question Answering:** SQuAD dataset

**Token budgets:**
- 512 tokens (short context)
- 1024 tokens (medium context)
- 2048 tokens (long context)

**Baselines:**
1. **Raw tokenization** - standard BPE/WordPiece
2. **Hybrid tokenization (FGT)** - mix of raw + glyph tokens

**Metrics:**
1. **Source text fitted:** number of characters/words that fit in token budget
2. **Summarization:** ROUGE-1, ROUGE-2, ROUGE-L
3. **QA:** Exact Match (EM), F1 score

### Protocol

**Step 1: Prepare datasets**
```python
from datasets import load_dataset

# Load XSum
xsum = load_dataset('xsum')

# Load SQuAD
squad = load_dataset('squad')
```

**Step 2: Measure source text capacity**
```python
def measure_capacity(texts, tokenizer, max_tokens):
    """Measure how much source text fits in token budget."""
    capacities = []
    for text in texts:
        encoded = tokenizer.encode(text, max_length=max_tokens, truncation=True)
        decoded = tokenizer.decode(encoded)
        capacities.append(len(decoded))  # character count
    return {
        'mean_chars': np.mean(capacities),
        'median_chars': np.median(capacities),
        'std_chars': np.std(capacities)
    }

# Compare raw vs. hybrid
raw_capacity = measure_capacity(xsum['test']['document'], raw_tokenizer, 512)
hybrid_capacity = measure_capacity(xsum['test']['document'], hybrid_tokenizer, 512)

expansion_factor = hybrid_capacity['mean_chars'] / raw_capacity['mean_chars']
print(f"Hybrid fits {expansion_factor:.2f}x more text in 512 tokens")
```

**Step 3: Evaluate summarization**
```python
from transformers import pipeline
from rouge_score import rouge_scorer

# Load summarization model (e.g., BART)
summarizer = pipeline('summarization', model='facebook/bart-base')

# Tokenize documents with different approaches
def summarize_with_tokenizer(documents, summaries, tokenizer, max_input=512):
    results = []
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    for doc, ref_summary in zip(documents, summaries):
        # Tokenize input
        encoded = tokenizer.encode(doc, max_length=max_input, truncation=True)
        input_text = tokenizer.decode(encoded)

        # Generate summary
        pred_summary = summarizer(input_text, max_length=100)[0]['summary_text']

        # Compute ROUGE
        scores = scorer.score(ref_summary, pred_summary)
        results.append(scores)

    return aggregate_rouge(results)

# Compare
raw_results = summarize_with_tokenizer(xsum['test']['document'][:1000],
                                       xsum['test']['summary'][:1000],
                                       raw_tokenizer, 512)
hybrid_results = summarize_with_tokenizer(xsum['test']['document'][:1000],
                                          xsum['test']['summary'][:1000],
                                          hybrid_tokenizer, 512)

print(f"Raw ROUGE-L: {raw_results['rougeL']:.4f}")
print(f"Hybrid ROUGE-L: {hybrid_results['rougeL']:.4f}")
print(f"Improvement: {(hybrid_results['rougeL'] - raw_results['rougeL']) * 100:.2f}%")
```

**Step 4: Evaluate QA**
```python
from transformers import pipeline

qa_pipeline = pipeline('question-answering', model='bert-base-uncased')

def evaluate_qa(contexts, questions, answers, tokenizer, max_context=512):
    em_scores = []
    f1_scores = []

    for context, question, answer in zip(contexts, questions, answers):
        # Tokenize context
        encoded = tokenizer.encode(context, max_length=max_context, truncation=True)
        context_text = tokenizer.decode(encoded)

        # Get prediction
        pred = qa_pipeline(question=question, context=context_text)

        # Compute metrics
        em = int(pred['answer'].strip() == answer['text'][0].strip())
        f1 = compute_f1(pred['answer'], answer['text'][0])

        em_scores.append(em)
        f1_scores.append(f1)

    return {
        'exact_match': np.mean(em_scores),
        'f1': np.mean(f1_scores)
    }

# Compare
raw_qa = evaluate_qa(squad['validation']['context'][:1000],
                     squad['validation']['question'][:1000],
                     squad['validation']['answers'][:1000],
                     raw_tokenizer, 512)
hybrid_qa = evaluate_qa(squad['validation']['context'][:1000],
                        squad['validation']['question'][:1000],
                        squad['validation']['answers'][:1000],
                        hybrid_tokenizer, 512)

print(f"Raw EM: {raw_qa['exact_match']:.4f}")
print(f"Hybrid EM: {hybrid_qa['exact_match']:.4f}")
```

### Expected Results

**Capacity (512-token budget):**
| Tokenizer | Mean Chars Fitted | Expansion Factor |
|-----------|-------------------|------------------|
| Raw | ~2,000 | 1.0x |
| Hybrid | ~5,000-8,000 | 2.5-4.0x |

**Summarization (ROUGE-L):**
| Tokenizer | 512 tokens | 1024 tokens |
|-----------|-----------|-------------|
| Raw | 0.35 | 0.40 |
| Hybrid | 0.38 (+8%) | 0.42 (+5%) |

**QA (Exact Match):**
| Tokenizer | 512 tokens | 1024 tokens |
|-----------|-----------|-------------|
| Raw | 0.70 | 0.75 |
| Hybrid | 0.735 (+5%) | 0.77 (+2.7%) |

### Analysis

**Key findings:**
- Summarization benefits most (needs broad context)
- QA benefits less (needs specific details)
- Gains diminish with larger budgets (less bottleneck)

### Deliverables

- [ ] Capacity comparison table
- [ ] ROUGE score improvements (Table 3)
- [ ] EM/F1 improvements
- [ ] Analysis by task type

---

## Experiment 3: Cross-Lingual Retrieval

### Goal
Demonstrate that glyph IDs act as language-agnostic anchors for cross-lingual phrase retrieval.

### Hypothesis
Glyph-based retrieval achieves 13-7 percentage point gains over embedding-only retrieval (Recall@10 and Recall@50).

### Setup

**Phrase index:**
- 50,000 English phrases
- 30,000 Chinese phrases
- 20,000 Spanish phrases
- Mixed from parallel and comparable corpora

**Query sets:**
- EN → ZH (500 queries)
- ZH → ES (500 queries)
- ES → EN (500 queries)

**Methods:**
1. **Glyph-based:** Match by cluster ID (exact match)
2. **Embedding-only:** Cosine similarity in embedding space (threshold > 0.8)

**Metrics:**
- Recall@10: % of queries with correct answer in top-10
- Recall@50: % of queries with correct answer in top-50

### Protocol

**Step 1: Build multilingual phrase index**
```python
# Build FGT tape from multilingual corpus
python scripts/build_tape.py \
  --input data/multilingual_phrases.jsonl \
  --output models/multilingual_tape.db \
  --num-clusters 20000 \
  --embedding-model LaBSE

# Index phrases by glyph and embedding
python scripts/index_phrases.py \
  --tape models/multilingual_tape.db \
  --output models/phrase_index.faiss
```

**Step 2: Prepare query set**
```python
# Create cross-lingual query pairs
# Format: {query_text, query_lang, target_text, target_lang}

queries = [
    {
        'query': 'Can you send me that file?',
        'query_lang': 'en',
        'target': '你能发给我那个文件吗？',
        'target_lang': 'zh'
    },
    # ... 500 pairs per language combination
]
```

**Step 3: Glyph-based retrieval**
```python
def glyph_retrieval(query_text, index, tape, target_lang, k=10):
    """Retrieve via glyph ID matching."""
    # Encode query to glyph
    query_glyph = encode_to_glyph(query_text, tape)

    # Find all phrases with same glyph in target language
    cluster_id = tape.glyph_to_cluster(query_glyph)
    candidates = index.get_phrases_by_cluster_and_lang(cluster_id, target_lang)

    # Rank by embedding similarity within cluster
    query_emb = embed(query_text)
    ranked = rank_by_similarity(query_emb, candidates)

    return ranked[:k]

# Evaluate
results = []
for query in queries:
    retrieved = glyph_retrieval(query['query'], index, tape, query['target_lang'], k=50)

    # Check if target is in top-K
    recall_at_10 = int(query['target'] in [r['text'] for r in retrieved[:10]])
    recall_at_50 = int(query['target'] in [r['text'] for r in retrieved[:50]])

    results.append({
        'recall_at_10': recall_at_10,
        'recall_at_50': recall_at_50
    })

glyph_r10 = np.mean([r['recall_at_10'] for r in results])
glyph_r50 = np.mean([r['recall_at_50'] for r in results])
```

**Step 4: Embedding-only retrieval (baseline)**
```python
def embedding_retrieval(query_text, index, target_lang, k=10):
    """Retrieve via embedding similarity only."""
    # Embed query
    query_emb = embed(query_text)

    # Nearest neighbor search in target language
    candidates = index.get_phrases_by_lang(target_lang)
    ranked = rank_by_similarity(query_emb, candidates)

    return ranked[:k]

# Evaluate
emb_results = []
for query in queries:
    retrieved = embedding_retrieval(query['query'], index, query['target_lang'], k=50)

    recall_at_10 = int(query['target'] in [r['text'] for r in retrieved[:10]])
    recall_at_50 = int(query['target'] in [r['text'] for r in retrieved[:50]])

    emb_results.append({
        'recall_at_10': recall_at_10,
        'recall_at_50': recall_at_50
    })

emb_r10 = np.mean([r['recall_at_10'] for r in emb_results])
emb_r50 = np.mean([r['recall_at_50'] for r in emb_results])
```

**Step 5: Compare**
```python
print("Glyph-based retrieval:")
print(f"  Recall@10: {glyph_r10:.4f}")
print(f"  Recall@50: {glyph_r50:.4f}")

print("Embedding-only retrieval:")
print(f"  Recall@10: {emb_r10:.4f}")
print(f"  Recall@50: {emb_r50:.4f}")

print("Improvements:")
print(f"  Recall@10: +{(glyph_r10 - emb_r10) * 100:.1f} pp")
print(f"  Recall@50: +{(glyph_r50 - emb_r50) * 100:.1f} pp")
```

### Expected Results

| Method | Recall@10 | Recall@50 |
|--------|-----------|-----------|
| Glyph-based | 0.78 | 0.91 |
| Embedding-only | 0.65 | 0.84 |
| **Improvement** | **+13 pp** | **+7 pp** |

**Per language pair:**
| Pair | Glyph R@10 | Embedding R@10 | Gain |
|------|------------|----------------|------|
| EN→ZH | 0.80 | 0.67 | +13 pp |
| ZH→ES | 0.75 | 0.62 | +13 pp |
| ES→EN | 0.79 | 0.66 | +13 pp |

### Analysis

**Key findings:**
- Glyphs provide hard anchors vs. soft embeddings
- Biggest gains in idiomatic/cultural phrases
- Embedding similarity still useful within clusters for ranking

### Deliverables

- [ ] Retrieval results table (Table 4)
- [ ] Per-language-pair breakdown
- [ ] Example retrievals (good and bad)
- [ ] Error analysis

---

## Experiment 4: Cluster Quality

### Goal
Assess the coherence and interpretability of phrase families.

### Hypothesis
72% of clusters are coherent (human eval 4-5/5), with avg intra-cluster similarity > 0.75.

### Setup

**Metrics:**
1. **Quantitative:**
   - Intra-cluster cosine similarity (avg within cluster)
   - Silhouette score (cluster separation)
2. **Qualitative:**
   - Human evaluation: rate 100 random clusters on coherence (1-5 scale)

**Sample:**
- 100 random clusters
- Stratified by size (small/medium/large)

### Protocol

**Step 1: Compute quantitative metrics**
```python
from sklearn.metrics import silhouette_score

def evaluate_cluster_quality(phrases, cluster_labels, embeddings):
    """Compute cluster quality metrics."""
    # Silhouette score
    silhouette = silhouette_score(embeddings, cluster_labels)

    # Intra-cluster similarity
    intra_sims = []
    for cluster_id in np.unique(cluster_labels):
        cluster_embs = embeddings[cluster_labels == cluster_id]
        # Pairwise cosine similarity
        sims = cosine_similarity(cluster_embs)
        # Average (excluding diagonal)
        avg_sim = (sims.sum() - len(sims)) / (len(sims) * (len(sims) - 1))
        intra_sims.append(avg_sim)

    return {
        'silhouette': silhouette,
        'mean_intra_cluster_sim': np.mean(intra_sims),
        'median_intra_cluster_sim': np.median(intra_sims),
        'std_intra_cluster_sim': np.std(intra_sims)
    }

metrics = evaluate_cluster_quality(phrases, cluster_labels, embeddings)
print(json.dumps(metrics, indent=2))
```

**Step 2: Sample clusters for human eval**
```python
# Sample 100 clusters stratified by size
small = [c for c in clusters if len(c) < 20]
medium = [c for c in clusters if 20 <= len(c) < 100]
large = [c for c in clusters if len(c) >= 100]

sample = (
    random.sample(small, 33) +
    random.sample(medium, 34) +
    random.sample(large, 33)
)

# Export for annotation
with open('human_eval_clusters.jsonl', 'w') as f:
    for cluster in sample:
        json.dump({
            'cluster_id': cluster.id,
            'size': len(cluster.phrases),
            'examples': random.sample(cluster.phrases, min(10, len(cluster.phrases))),
            'languages': cluster.language_distribution
        }, f)
        f.write('\n')
```

**Step 3: Human evaluation**

**Annotation guidelines:**
```
Rate each cluster on coherence (1-5):

5 - Highly coherent: All examples express the same intent/meaning
4 - Mostly coherent: 80%+ examples are related
3 - Mixed: Some related, some unrelated
2 - Mostly incoherent: 80%+ examples are unrelated
1 - Completely incoherent: No discernible pattern

Consider:
- Semantic similarity across examples
- Cross-lingual consistency (if multilingual)
- Presence of outliers
```

**Step 4: Aggregate human eval**
```python
import pandas as pd

# Load annotations
annotations = pd.read_json('human_eval_results.jsonl', lines=True)

# Compute agreement (if multiple annotators)
from sklearn.metrics import cohen_kappa_score
kappa = cohen_kappa_score(annotations['annotator1'], annotations['annotator2'])
print(f"Inter-annotator agreement (Cohen's kappa): {kappa:.3f}")

# Aggregate scores
coherence_dist = annotations['coherence_score'].value_counts(normalize=True)
print("\nCoherence distribution:")
print(coherence_dist)

coherent_pct = annotations[annotations['coherence_score'] >= 4].shape[0] / len(annotations)
print(f"\n% coherent (4-5): {coherent_pct * 100:.1f}%")
```

### Expected Results

**Quantitative:**
- Avg intra-cluster similarity: 0.76
- Silhouette score: 0.58
- Distribution: see histogram

**Qualitative:**
- 72% rated 4-5 (coherent)
- 20% rated 3 (mixed)
- 8% rated 1-2 (incoherent)

**Incoherent clusters arise from:**
- Polysemy (same phrase, different meanings)
- Embedding noise
- Under-clustering (K too small)

### Deliverables

- [ ] Cluster quality metrics table
- [ ] Coherence distribution histogram (Figure 6)
- [ ] Example coherent and incoherent clusters
- [ ] Error analysis

---

## Experiment 5: Fractal Address Space Quality

### Goal
Evaluate whether fractal addresses preserve semantic neighborhoods.

### Hypothesis
68% of embedding neighbors are also fractal neighbors (K=10).

### Setup

**Metric:**
- Neighborhood preservation: For each cluster, compute % of K-nearest neighbors in embedding space that are also spatial neighbors in fractal space.

**K values:** 10, 20, 50

**Baseline:** Random addressing (shuffle addresses)

### Protocol

**Step 1: Compute embedding neighbors**
```python
from sklearn.neighbors import NearestNeighbors

# Build KNN index in embedding space
knn = NearestNeighbors(n_neighbors=51, metric='cosine')
knn.fit(cluster_centroids)

# For each cluster, find K nearest neighbors in embedding space
emb_neighbors = {}
for i, centroid in enumerate(cluster_centroids):
    distances, indices = knn.kneighbors([centroid], n_neighbors=51)
    emb_neighbors[i] = indices[0][1:]  # Exclude self
```

**Step 2: Compute fractal neighbors**
```python
def get_fractal_neighbors(cluster_id, tape, k=10):
    """Get K nearest neighbors in fractal space."""
    # Get fractal address of cluster
    address = tape.get_address(cluster_id)

    # Find clusters with similar addresses (edit distance)
    candidates = []
    for other_id, other_address in tape.get_all_addresses():
        if other_id != cluster_id:
            dist = edit_distance(address, other_address)
            candidates.append((other_id, dist))

    # Sort by distance, take top-K
    candidates.sort(key=lambda x: x[1])
    return [c[0] for c in candidates[:k]]

fractal_neighbors = {}
for cluster_id in range(num_clusters):
    fractal_neighbors[cluster_id] = get_fractal_neighbors(cluster_id, tape, k=50)
```

**Step 3: Measure preservation**
```python
def measure_preservation(emb_neighbors, fractal_neighbors, k=10):
    """Measure % of emb neighbors that are also fractal neighbors."""
    preservation_scores = []

    for cluster_id in range(len(emb_neighbors)):
        emb_k = set(emb_neighbors[cluster_id][:k])
        fractal_k = set(fractal_neighbors[cluster_id][:k])

        overlap = len(emb_k & fractal_k)
        preservation = overlap / k
        preservation_scores.append(preservation)

    return {
        'mean': np.mean(preservation_scores),
        'median': np.median(preservation_scores),
        'std': np.std(preservation_scores)
    }

# Compute for different K values
for k in [10, 20, 50]:
    results = measure_preservation(emb_neighbors, fractal_neighbors, k=k)
    print(f"K={k}: {results['mean']:.3f} preservation")

# Baseline: random addressing
random_neighbors = {i: random.sample(range(num_clusters), 50) for i in range(num_clusters)}
random_results = measure_preservation(emb_neighbors, random_neighbors, k=10)
print(f"Random baseline (K=10): {random_results['mean']:.3f}")
```

### Expected Results

| K | Preservation | Random Baseline |
|---|--------------|-----------------|
| 10 | 0.68 | 0.10 |
| 20 | 0.62 | 0.15 |
| 50 | 0.54 | 0.20 |

**Analysis:**
- Fractal addressing preserves coarse semantic structure
- Fine-grained details lost (2D projection trade-off)
- Significantly better than random

### Deliverables

- [ ] Neighborhood preservation results
- [ ] Preservation vs. K plot
- [ ] Comparison to random baseline
- [ ] Visualization of fractal neighborhoods

---

## Ablation Studies

### Ablation 1: Glyph Encoding Schemes

**Variants:**
1. Frequency-aware (current)
2. Random assignment
3. Semantic clustering-based

**Protocol:**
```python
# Build tapes with different glyph schemes
for scheme in ['frequency', 'random', 'semantic']:
    build_tape(
        data='data/corpus.jsonl',
        glyph_scheme=scheme,
        output=f'models/tape_{scheme}.db'
    )

    # Evaluate compression
    compression = evaluate_compression(f'models/tape_{scheme}.db', test_data)
    print(f"{scheme}: {compression['ratio']:.3f}")
```

**Expected results:**
- Frequency-aware: 0.65 compression
- Random: 0.58 compression (-12%)
- Semantic: 0.62 compression, better interpretability

---

### Ablation 2: Fractal Depth

**Variants:**
- Depth 4, 6, 8, 10

**Protocol:**
```python
for depth in [4, 6, 8, 10]:
    build_tape(
        data='data/corpus.jsonl',
        fractal_depth=depth,
        output=f'models/tape_depth{depth}.db'
    )

    # Evaluate neighborhood preservation
    preservation = evaluate_preservation(f'models/tape_depth{depth}.db')
    print(f"Depth {depth}: {preservation['k10']:.3f}")
```

**Expected results:**
- Depth 4: Too coarse, collisions
- Depth 6-8: Optimal
- Depth 10: Too sparse, many empty cells

---

### Ablation 3: Clustering Algorithms

**Variants:**
- MiniBatchKMeans (current)
- HDBSCAN
- Agglomerative clustering
- BIRCH

**Protocol:**
```python
for algorithm in ['minibatch_kmeans', 'hdbscan', 'agglomerative', 'birch']:
    clusters = run_clustering(
        embeddings=embeddings,
        algorithm=algorithm,
        params=get_default_params(algorithm)
    )

    # Evaluate quality
    quality = evaluate_cluster_quality(clusters)
    print(f"{algorithm}: silhouette={quality['silhouette']:.3f}")
```

**Expected results:**
- MiniBatchKMeans: Fast, reasonable quality
- HDBSCAN: Better quality, slower, variable K
- Agglomerative: Good quality, doesn't scale
- BIRCH: Fast, lower quality

---

## Reproducibility Checklist

To ensure reproducibility of all experiments:

- [ ] Fix random seeds (Python, NumPy, PyTorch)
- [ ] Document exact library versions (requirements.txt)
- [ ] Save model checkpoints and configs
- [ ] Log all hyperparameters
- [ ] Save raw experimental results (JSON/CSV)
- [ ] Version control experiment scripts
- [ ] Document hardware (GPU model, RAM, CPU)
- [ ] Provide sample data for testing
- [ ] Include data preprocessing scripts
- [ ] Document annotation guidelines (human eval)

---

## Timeline

**Week 1:**
- Dataset preparation
- Build FGT tapes for all domains
- Run Experiment 1 (Compression)

**Week 2:**
- Run Experiment 2 (Context Extension)
- Run Experiment 3 (Cross-Lingual Retrieval)

**Week 3:**
- Run Experiment 4 (Cluster Quality)
  - Human evaluation in parallel
- Run Experiment 5 (Fractal Quality)

**Week 4:**
- Run all ablation studies
- Generate figures and tables
- Document results

---

## Resources Required

**Compute:**
- GPU: NVIDIA A100 or V100 (16GB+ VRAM)
- CPU: 16+ cores
- RAM: 64GB+
- Storage: 500GB+

**Data:**
- OpenSubtitles (100K-1M phrases)
- Wikipedia dumps (EN/ZH/ES)
- Customer support logs (if available)
- XSum, SQuAD datasets

**Human annotation:**
- 2-3 annotators
- ~4-6 hours for 100 clusters
- Compensation: $15-20/hour

---

## Success Criteria

**Experiments succeed if:**

1. ✅ **Exp 1:** Compression ≥ 55%, BERTScore ≥ 0.90
2. ✅ **Exp 2:** Context extension ≥ 2.5x, ROUGE improvement ≥ 5%
3. ✅ **Exp 3:** Glyph retrieval ≥ 10pp improvement over baseline
4. ✅ **Exp 4:** ≥ 70% clusters rated coherent (4-5/5)
5. ✅ **Exp 5:** Neighborhood preservation ≥ 60% (K=10)

If any experiment fails to meet criteria:
- Analyze failure modes
- Adjust hyperparameters
- Re-run with improved setup
- Document honestly in paper

---

**Good luck running experiments! Document everything carefully for reproducibility.**
