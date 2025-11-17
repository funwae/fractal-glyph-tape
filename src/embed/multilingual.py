"""Multilingual embedding support for cross-lingual phrase clustering."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


class MultilingualEmbedder:
    """
    Generate multilingual embeddings for cross-lingual clustering.

    Uses multilingual sentence embedding models (e.g., multilingual-e5, LaBSE)
    to create embeddings that work across languages.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        device: str = "cpu",
    ):
        """
        Initialize multilingual embedder.

        Args:
            model_name: Name of multilingual embedding model
            device: Device to run on (cpu, cuda)
        """
        self.model_name = model_name
        self.device = device
        self._model = None

    def _load_model(self):
        """Lazy load the embedding model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name, device=self.device)
                print(f"Loaded multilingual model: {self.model_name}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers required. Install with: "
                    "pip install sentence-transformers"
                )

    def embed(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Generate embeddings for texts.

        Args:
            texts: List of text phrases
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar

        Returns:
            Array of embeddings with shape (n_texts, embedding_dim)
        """
        self._load_model()

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

        return embeddings

    def embed_single(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        return self.embed([text], show_progress=False)[0]

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Cosine similarity score
        """
        emb1 = self.embed_single(text1)
        emb2 = self.embed_single(text2)

        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

        return float(similarity)

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        self._load_model()
        return self._model.get_sentence_embedding_dimension()

    def __repr__(self) -> str:
        """String representation."""
        return f"MultilingualEmbedder(model={self.model_name}, device={self.device})"


class LanguageDetector:
    """
    Detect language of text using langdetect or fasttext.

    Supports detection for major languages to track cluster language distribution.
    """

    def __init__(self):
        """Initialize language detector."""
        self._detector = None

    def _load_detector(self):
        """Lazy load language detector."""
        if self._detector is None:
            try:
                import langdetect

                # Set seed for reproducibility
                langdetect.DetectorFactory.seed = 0
                self._detector = "langdetect"
                print("Loaded langdetect for language detection")
            except ImportError:
                print(
                    "Warning: langdetect not available. Install with: "
                    "pip install langdetect"
                )
                self._detector = "fallback"

    def detect(self, text: str) -> str:
        """
        Detect language of text.

        Args:
            text: Input text

        Returns:
            ISO 639-1 language code (e.g., 'en', 'zh', 'es')
        """
        self._load_detector()

        if self._detector == "langdetect":
            try:
                import langdetect

                lang = langdetect.detect(text)
                return lang
            except Exception:
                return "unknown"
        else:
            # Fallback: simple heuristic
            return self._simple_detect(text)

    def _simple_detect(self, text: str) -> str:
        """
        Simple fallback language detection.

        Uses character ranges to detect common languages.

        Args:
            text: Input text

        Returns:
            Language code
        """
        # Count character types
        has_cjk = any("\u4e00" <= c <= "\u9fff" for c in text)
        has_arabic_script = any("\u0600" <= c <= "\u06ff" for c in text)
        has_cyrillic = any("\u0400" <= c <= "\u04ff" for c in text)

        if has_cjk:
            return "zh"  # Assuming Chinese for CJK
        elif has_arabic_script:
            return "ar"
        elif has_cyrillic:
            return "ru"
        else:
            return "en"  # Default to English

    def detect_with_confidence(self, text: str) -> Tuple[str, float]:
        """
        Detect language with confidence score.

        Args:
            text: Input text

        Returns:
            Tuple of (language_code, confidence)
        """
        self._load_detector()

        if self._detector == "langdetect":
            try:
                import langdetect

                probabilities = langdetect.detect_langs(text)
                if probabilities:
                    best = probabilities[0]
                    return str(best.lang), float(best.prob)
            except Exception:
                pass

        # Fallback
        lang = self.detect(text)
        return lang, 0.5  # Low confidence for fallback

    def detect_batch(self, texts: List[str]) -> List[str]:
        """
        Detect languages for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of language codes
        """
        return [self.detect(text) for text in texts]


class MultilingualClusterAnalyzer:
    """
    Analyze language distribution in clusters for cross-lingual properties.
    """

    def __init__(self, cluster_metadata: Dict):
        """
        Initialize analyzer.

        Args:
            cluster_metadata: Dictionary of cluster metadata
        """
        self.cluster_metadata = cluster_metadata
        self.language_detector = LanguageDetector()

    def analyze_cluster_languages(self, cluster_id: str) -> Dict:
        """
        Analyze language distribution in a cluster.

        Args:
            cluster_id: Cluster identifier

        Returns:
            Dictionary with language statistics
        """
        meta = self.cluster_metadata.get(cluster_id, {})
        phrases = meta.get("example_phrases", [])

        if not phrases:
            return {
                "cluster_id": cluster_id,
                "total_phrases": 0,
                "languages": {},
                "is_multilingual": False,
                "entropy": 0.0,
            }

        # Detect languages
        languages = self.language_detector.detect_batch(phrases)

        # Count languages
        lang_counts = {}
        for lang in languages:
            lang_counts[lang] = lang_counts.get(lang, 0) + 1

        # Compute statistics
        total = len(languages)
        lang_probs = {lang: count / total for lang, count in lang_counts.items()}

        # Compute entropy
        entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in lang_probs.values())

        return {
            "cluster_id": cluster_id,
            "total_phrases": total,
            "languages": lang_counts,
            "language_probabilities": lang_probs,
            "is_multilingual": len(lang_counts) > 1,
            "entropy": float(entropy),
            "dominant_language": max(lang_counts.items(), key=lambda x: x[1])[0]
            if lang_counts
            else "unknown",
        }

    def analyze_all_clusters(self) -> Dict:
        """
        Analyze language distribution across all clusters.

        Returns:
            Dictionary with global statistics
        """
        cluster_stats = []
        multilingual_count = 0
        total_entropy = 0.0

        for cluster_id in self.cluster_metadata.keys():
            stats = self.analyze_cluster_languages(cluster_id)
            cluster_stats.append(stats)

            if stats["is_multilingual"]:
                multilingual_count += 1

            total_entropy += stats["entropy"]

        total_clusters = len(cluster_stats)
        avg_entropy = total_entropy / total_clusters if total_clusters > 0 else 0

        # Count phrases per language
        global_lang_counts = {}
        for stats in cluster_stats:
            for lang, count in stats["languages"].items():
                global_lang_counts[lang] = global_lang_counts.get(lang, 0) + count

        return {
            "total_clusters": total_clusters,
            "multilingual_clusters": multilingual_count,
            "multilingual_percentage": (multilingual_count / total_clusters * 100)
            if total_clusters > 0
            else 0,
            "average_entropy": avg_entropy,
            "global_language_distribution": global_lang_counts,
            "cluster_stats": cluster_stats,
        }


if __name__ == "__main__":
    # Example usage
    print("Multilingual Embedding Example")
    print("=" * 60)

    # Test embedder
    print("\nTesting multilingual embedder...")
    embedder = MultilingualEmbedder()

    # Test phrases in different languages
    phrases = [
        "Can you send me that file?",  # English
        "你能发给我那个文件吗？",  # Chinese
        "¿Puedes enviarme ese archivo?",  # Spanish
    ]

    print(f"\nEmbedding {len(phrases)} phrases...")
    embeddings = embedder.embed(phrases, show_progress=False)
    print(f"Embeddings shape: {embeddings.shape}")

    # Test similarity
    print("\nCross-lingual similarity:")
    sim_en_zh = embedder.compute_similarity(phrases[0], phrases[1])
    sim_en_es = embedder.compute_similarity(phrases[0], phrases[2])
    print(f"  EN-ZH: {sim_en_zh:.3f}")
    print(f"  EN-ES: {sim_en_es:.3f}")

    # Test language detection
    print("\nTesting language detection...")
    detector = LanguageDetector()

    for phrase in phrases:
        lang, conf = detector.detect_with_confidence(phrase)
        print(f"  '{phrase[:30]}...' → {lang} (conf: {conf:.2f})")
