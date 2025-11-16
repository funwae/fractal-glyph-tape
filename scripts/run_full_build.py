#!/usr/bin/env python3
"""
Full FGT pipeline orchestrator.
Usage: python scripts/run_full_build.py --config configs/demo.yaml
"""

import sys
import argparse
import yaml
from pathlib import Path
from loguru import logger


def run_full_build(config_path: str):
    """
    Run the complete FGT build pipeline.

    Args:
        config_path: Path to configuration YAML file
    """
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add("logs/fgt_build.log", level="DEBUG", rotation="10 MB")

    logger.info("=" * 80)
    logger.info("FRACTAL GLYPH TAPE - FULL BUILD PIPELINE")
    logger.info("=" * 80)

    # Load config
    config_path = Path(config_path)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    logger.info(f"Project: {cfg['project_name']}")
    logger.info(f"Config: {config_path}")
    logger.info("")

    try:
        # Import modules
        from ingest import PhraseExtractor
        from embed import PhraseEmbedder
        from cluster import PhraseClusterer, ClusterMetadata
        from glyph import GlyphManager
        from tape import TapeBuilder

        # Step 1: Ingest phrases
        logger.info("=" * 80)
        logger.info("STEP 1: PHRASE INGESTION")
        logger.info("=" * 80)
        extractor = PhraseExtractor(cfg["ingest"])
        extractor.extract_phrases()
        logger.info("✓ Phrase ingestion complete\n")

        # Step 2: Embed phrases
        logger.info("=" * 80)
        logger.info("STEP 2: PHRASE EMBEDDING")
        logger.info("=" * 80)
        embedder = PhraseEmbedder(cfg["embed"])
        index_data = embedder.embed_phrases_from_file(
            cfg["ingest"]["output_path"],
            cfg["embed"]["output_path"]
        )
        logger.info("✓ Phrase embedding complete\n")

        # Step 3: Cluster embeddings
        logger.info("=" * 80)
        logger.info("STEP 3: CLUSTERING")
        logger.info("=" * 80)
        embeddings = embedder.load_embeddings(cfg["embed"]["output_path"])
        clusterer = PhraseClusterer(cfg["cluster"])
        cluster_stats = clusterer.cluster_embeddings(embeddings)
        clusterer.save_results(cfg["cluster"]["output_path"])
        logger.info("✓ Clustering complete\n")

        # Step 4: Extract cluster metadata
        logger.info("=" * 80)
        logger.info("STEP 4: CLUSTER METADATA EXTRACTION")
        logger.info("=" * 80)
        metadata_extractor = ClusterMetadata(cfg)
        cluster_metadata = metadata_extractor.extract_metadata(
            cfg["ingest"]["output_path"],
            clusterer.labels,
            clusterer.centroids,
            embeddings
        )
        metadata_path = Path(cfg["cluster"]["output_path"]) / "metadata.json"
        metadata_extractor.save_metadata(str(metadata_path))
        logger.info("✓ Metadata extraction complete\n")

        # Step 5: Assign glyph IDs
        logger.info("=" * 80)
        logger.info("STEP 5: GLYPH ID ASSIGNMENT")
        logger.info("=" * 80)
        glyph_manager = GlyphManager(cfg["glyph"])
        glyph_manager.assign_glyphs(len(clusterer.centroids))
        glyph_manager.save_mapping(cfg["glyph"]["output_path"])
        logger.info("✓ Glyph assignment complete\n")

        # Step 6: Build fractal tape
        logger.info("=" * 80)
        logger.info("STEP 6: FRACTAL TAPE CONSTRUCTION")
        logger.info("=" * 80)
        tape_builder = TapeBuilder(cfg["tape"])
        tape_db_path = tape_builder.build_tape(
            clusterer.centroids,
            glyph_manager.cluster_to_glyph,
            cluster_metadata
        )
        logger.info("✓ Tape construction complete\n")

        # Final summary
        logger.info("=" * 80)
        logger.info("BUILD COMPLETE!")
        logger.info("=" * 80)
        logger.info(f"Total phrases: {index_data['total_phrases']:,}")
        logger.info(f"Embedding dimension: {index_data['embedding_dim']}")
        logger.info(f"Number of clusters: {cluster_stats['n_clusters']:,}")
        logger.info(f"Tape database: {tape_db_path}")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"Build failed with error: {e}")
        logger.exception(e)
        return 1


def main():
    parser = argparse.ArgumentParser(description="Run full FGT build pipeline")
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    args = parser.parse_args()

    exit_code = run_full_build(args.config)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
