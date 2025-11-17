"""Cross-lingual clustering with language tracking."""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from ..embed.multilingual import LanguageDetector, MultilingualEmbedder


class CrossLingualClusterer:
    """
    Cluster phrases across languages to create cross-lingual phrase families.

    Uses multilingual embeddings to group semantically similar phrases
    regardless of language.
    """

    def __init__(
        self,
        n_clusters: int = 10000,
        embedder: Optional[MultilingualEmbedder] = None,
        language_detector: Optional[LanguageDetector] = None,
        random_state: int = 42,
    ):
        """
        Initialize cross-lingual clusterer.

        Args:
            n_clusters: Number of clusters to create
            embedder: Multilingual embedder (creates default if None)
            language_detector: Language detector (creates default if None)
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state

        self.embedder = embedder or MultilingualEmbedder()
        self.language_detector = language_detector or LanguageDetector()

        self.clusterer = None
        self.cluster_centers = None
        self.cluster_metadata = {}

    def fit(
        self,
        phrases: List[str],
        batch_size: int = 1000,
        show_progress: bool = True,
    ) -> "CrossLingualClusterer":
        """
        Fit clusterer on phrases.

        Args:
            phrases: List of phrases to cluster
            batch_size: Batch size for embedding
            show_progress: Whether to show progress

        Returns:
            Self for chaining
        """
        print(f"\nCross-Lingual Clustering")
        print("=" * 60)
        print(f"Phrases: {len(phrases)}")
        print(f"Target clusters: {self.n_clusters}")
        print(f"Embedder: {self.embedder}")

        # Generate embeddings
        print("\nGenerating multilingual embeddings...")
        embeddings = self.embedder.embed(
            phrases,
            batch_size=batch_size,
            show_progress=show_progress,
        )

        print(f"Embeddings shape: {embeddings.shape}")

        # Detect languages
        print("\nDetecting languages...")
        languages = self.language_detector.detect_batch(phrases)

        # Count languages
        lang_counts = defaultdict(int)
        for lang in languages:
            lang_counts[lang] += 1

        print(f"Language distribution:")
        for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1]):
            print(f"  {lang}: {count} phrases ({count/len(phrases)*100:.1f}%)")

        # Cluster
        print(f"\nClustering into {self.n_clusters} families...")
        self.clusterer = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            batch_size=batch_size,
            verbose=1 if show_progress else 0,
        )

        labels = self.clusterer.fit_predict(embeddings)
        self.cluster_centers = self.clusterer.cluster_centers_

        # Build cluster metadata
        print("\nBuilding cluster metadata...")
        self._build_metadata(phrases, languages, labels, embeddings)

        print("\nClustering complete!")
        return self

    def _build_metadata(
        self,
        phrases: List[str],
        languages: List[str],
        labels: np.ndarray,
        embeddings: np.ndarray,
    ):
        """
        Build metadata for each cluster.

        Args:
            phrases: Original phrases
            languages: Detected languages
            labels: Cluster labels
            embeddings: Phrase embeddings
        """
        # Group by cluster
        cluster_phrases = defaultdict(list)
        cluster_languages = defaultdict(list)
        cluster_embeddings = defaultdict(list)

        for phrase, lang, label, emb in zip(phrases, languages, labels, embeddings):
            cluster_phrases[label].append(phrase)
            cluster_languages[label].append(lang)
            cluster_embeddings[label].append(emb)

        # Create metadata for each cluster
        for cluster_id in range(self.n_clusters):
            if cluster_id not in cluster_phrases:
                # Empty cluster
                self.cluster_metadata[str(cluster_id)] = {
                    "size": 0,
                    "example_phrases": [],
                    "languages": {},
                    "is_multilingual": False,
                }
                continue

            phrases_in_cluster = cluster_phrases[cluster_id]
            langs_in_cluster = cluster_languages[cluster_id]
            embs_in_cluster = np.array(cluster_embeddings[cluster_id])

            # Count languages
            lang_counts = defaultdict(int)
            for lang in langs_in_cluster:
                lang_counts[lang] += 1

            # Find representative phrase (closest to centroid)
            center = self.cluster_centers[cluster_id]
            distances = np.linalg.norm(embs_in_cluster - center, axis=1)
            representative_idx = np.argmin(distances)
            representative_phrase = phrases_in_cluster[representative_idx]

            # Language statistics
            total = len(phrases_in_cluster)
            lang_probs = {lang: count / total for lang, count in lang_counts.items()}

            # Entropy
            entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in lang_probs.values())

            self.cluster_metadata[str(cluster_id)] = {
                "size": total,
                "representative_phrase": representative_phrase,
                "example_phrases": phrases_in_cluster[:100],  # Limit to 100
                "languages": dict(lang_counts),
                "language_probabilities": lang_probs,
                "is_multilingual": len(lang_counts) > 1,
                "entropy": float(entropy),
                "dominant_language": max(lang_counts.items(), key=lambda x: x[1])[0],
                "embedding_centroid": center.tolist(),
            }

    def predict(
        self,
        phrases: List[str],
        return_languages: bool = False,
    ) -> np.ndarray:
        """
        Predict cluster labels for new phrases.

        Args:
            phrases: List of phrases
            return_languages: Whether to also return detected languages

        Returns:
            Array of cluster labels, optionally with languages
        """
        if self.clusterer is None:
            raise ValueError("Clusterer not fitted. Call fit() first.")

        # Embed
        embeddings = self.embedder.embed(phrases, show_progress=False)

        # Predict
        labels = self.clusterer.predict(embeddings)

        if return_languages:
            languages = self.language_detector.detect_batch(phrases)
            return labels, languages

        return labels

    def get_cross_lingual_clusters(self, min_languages: int = 2) -> List[str]:
        """
        Get clusters that contain multiple languages.

        Args:
            min_languages: Minimum number of languages required

        Returns:
            List of cluster IDs
        """
        cross_lingual = []

        for cluster_id, meta in self.cluster_metadata.items():
            if len(meta["languages"]) >= min_languages:
                cross_lingual.append(cluster_id)

        return cross_lingual

    def get_cluster_examples_by_language(
        self,
        cluster_id: str,
        language: str,
        max_examples: int = 5,
    ) -> List[str]:
        """
        Get example phrases from a cluster in a specific language.

        Args:
            cluster_id: Cluster identifier
            language: Language code
            max_examples: Maximum number of examples

        Returns:
            List of example phrases
        """
        meta = self.cluster_metadata.get(cluster_id, {})
        all_phrases = meta.get("example_phrases", [])

        # Filter by language
        examples = []
        for phrase in all_phrases:
            detected_lang = self.language_detector.detect(phrase)
            if detected_lang == language:
                examples.append(phrase)
                if len(examples) >= max_examples:
                    break

        return examples

    def save(self, output_dir: Path):
        """
        Save clusterer state.

        Args:
            output_dir: Directory to save to
        """
        import json
        import pickle

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save cluster centers
        np.save(output_dir / "cluster_centers.npy", self.cluster_centers)

        # Save metadata
        with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.cluster_metadata, f, ensure_ascii=False, indent=2)

        # Save clusterer
        with open(output_dir / "clusterer.pkl", "wb") as f:
            pickle.dump(self.clusterer, f)

        print(f"Saved cross-lingual clusterer to {output_dir}")

    @classmethod
    def load(
        cls,
        input_dir: Path,
        embedder: Optional[MultilingualEmbedder] = None,
        language_detector: Optional[LanguageDetector] = None,
    ) -> "CrossLingualClusterer":
        """
        Load clusterer from disk.

        Args:
            input_dir: Directory to load from
            embedder: Embedder to use
            language_detector: Language detector to use

        Returns:
            Loaded clusterer
        """
        import json
        import pickle

        input_dir = Path(input_dir)

        # Create instance
        instance = cls(embedder=embedder, language_detector=language_detector)

        # Load cluster centers
        instance.cluster_centers = np.load(input_dir / "cluster_centers.npy")
        instance.n_clusters = len(instance.cluster_centers)

        # Load metadata
        with open(input_dir / "metadata.json", "r", encoding="utf-8") as f:
            instance.cluster_metadata = json.load(f)

        # Load clusterer
        with open(input_dir / "clusterer.pkl", "rb") as f:
            instance.clusterer = pickle.load(f)

        print(f"Loaded cross-lingual clusterer from {input_dir}")
        return instance

    def get_statistics(self) -> Dict:
        """
        Get clustering statistics.

        Returns:
            Dictionary with statistics
        """
        total_clusters = len(self.cluster_metadata)
        multilingual_count = sum(
            1 for meta in self.cluster_metadata.values() if meta.get("is_multilingual", False)
        )

        # Global language distribution
        global_langs = defaultdict(int)
        total_phrases = 0

        for meta in self.cluster_metadata.values():
            for lang, count in meta.get("languages", {}).items():
                global_langs[lang] += count
                total_phrases += count

        return {
            "total_clusters": total_clusters,
            "multilingual_clusters": multilingual_count,
            "multilingual_percentage": (multilingual_count / total_clusters * 100)
            if total_clusters > 0
            else 0,
            "global_language_distribution": dict(global_langs),
            "total_phrases": total_phrases,
        }


if __name__ == "__main__":
    # Example usage
    print("Cross-Lingual Clustering Example")
    print("=" * 60)

    # Create sample multilingual phrases
    phrases = [
        # English
        "Can you send me that file?",
        "Please email the document",
        "Share that file with me",
        # Chinese
        "你能发给我那个文件吗？",
        "请把文件发给我",
        # Spanish
        "¿Puedes enviarme ese archivo?",
        "Por favor envíame el documento",
    ]

    # Cluster
    clusterer = CrossLingualClusterer(n_clusters=3)
    clusterer.fit(phrases, show_progress=False)

    # Get statistics
    stats = clusterer.get_statistics()
    print(f"\nStatistics:")
    print(f"  Total clusters: {stats['total_clusters']}")
    print(f"  Multilingual clusters: {stats['multilingual_clusters']}")
    print(f"  Languages: {stats['global_language_distribution']}")

    # Find cross-lingual clusters
    cross_lingual = clusterer.get_cross_lingual_clusters()
    print(f"\nCross-lingual clusters: {cross_lingual}")
