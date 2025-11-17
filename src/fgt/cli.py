"""Command-line interface for Fractal Glyph Tape.

Usage:
    fgt build --config configs/demo.yaml
    fgt encode "Can you send me that file?"
    fgt decode "谷阜"
    fgt inspect-glyph 谷阜
"""

import sys
import typer
import yaml
from pathlib import Path
from typing import Optional
from loguru import logger

app = typer.Typer(help="Fractal Glyph Tape - Semantic compression and phrase memory")


@app.command()
def build(
    config: Path = typer.Option(..., "--config", help="Path to config YAML file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Run the full FGT build pipeline."""
    # Configure logging
    logger.remove()
    if verbose:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")

    logger.info(f"Building FGT from config: {config}")

    # Load config
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)

    # Import and run pipeline steps
    try:
        from ingest import PhraseExtractor
        from embed import PhraseEmbedder
        from cluster import PhraseClusterer, ClusterMetadata
        from glyph import GlyphManager
        from tape import TapeBuilder

        # Step 1: Ingest phrases
        logger.info("=" * 60)
        logger.info("Step 1: Ingesting phrases")
        logger.info("=" * 60)
        extractor = PhraseExtractor(cfg["ingest"])
        extractor.extract_phrases()

        # Step 2: Embed phrases
        logger.info("=" * 60)
        logger.info("Step 2: Embedding phrases")
        logger.info("=" * 60)
        embedder = PhraseEmbedder(cfg["embed"])
        embedder.embed_phrases_from_file(
            cfg["ingest"]["output_path"],
            cfg["embed"]["output_path"]
        )

        # Step 3: Load embeddings and cluster
        logger.info("=" * 60)
        logger.info("Step 3: Clustering embeddings")
        logger.info("=" * 60)
        embeddings = embedder.load_embeddings(cfg["embed"]["output_path"])
        clusterer = PhraseClusterer(cfg["cluster"])
        clusterer.cluster_embeddings(embeddings)
        clusterer.save_results(cfg["cluster"]["output_path"])

        # Step 4: Extract cluster metadata
        logger.info("=" * 60)
        logger.info("Step 4: Extracting cluster metadata")
        logger.info("=" * 60)
        metadata_extractor = ClusterMetadata(cfg)
        cluster_metadata = metadata_extractor.extract_metadata(
            cfg["ingest"]["output_path"],
            clusterer.labels,
            clusterer.centroids,
            embeddings
        )
        metadata_path = Path(cfg["cluster"]["output_path"]) / "metadata.json"
        metadata_extractor.save_metadata(str(metadata_path))

        # Step 5: Assign glyphs
        logger.info("=" * 60)
        logger.info("Step 5: Assigning glyph IDs")
        logger.info("=" * 60)
        glyph_manager = GlyphManager(cfg["glyph"])
        glyph_manager.assign_glyphs(cfg["cluster"]["n_clusters"])
        glyph_manager.save_mapping(cfg["glyph"]["output_path"])

        # Step 6: Build fractal tape
        logger.info("=" * 60)
        logger.info("Step 6: Building fractal tape")
        logger.info("=" * 60)
        tape_builder = TapeBuilder(cfg["tape"])
        tape_db_path = tape_builder.build_tape(
            clusterer.centroids,
            glyph_manager.cluster_to_glyph,
            cluster_metadata
        )

        logger.info("=" * 60)
        logger.info("FGT build complete!")
        logger.info(f"Tape database: {tape_db_path}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Build failed: {e}")
        raise typer.Exit(1)


@app.command()
def encode(
    text: str = typer.Argument(..., help="Text to encode as glyph representation"),
    tape_path: Optional[Path] = typer.Option("tape/v1/tape_index.db", "--tape", help="Path to tape storage"),
):
    """Encode text to glyph-coded representation."""
    from tape import TapeStorage
    from embed import PhraseEmbedder
    import numpy as np
    import yaml

    logger.info(f"Encoding: {text}")

    try:
        # Load embedder (you'll need to save config or use defaults)
        config_path = Path("configs/demo.yaml")
        if config_path.exists():
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
            embedder = PhraseEmbedder(cfg["embed"])
        else:
            logger.warning("Config not found, using default embedder")
            embedder = PhraseEmbedder({"model_name": "sentence-transformers/all-MiniLM-L6-v2"})

        # Embed the text
        embedding = embedder._embed_batch([text])[0]

        # Load tape and find nearest cluster
        with TapeStorage(str(tape_path)) as storage:
            storage.connect()
            # For now, just show that it works
            # In a full implementation, you'd search for nearest centroid
            logger.info("Tape loaded successfully")
            typer.echo(f"Text: {text}")
            typer.echo(f"Embedding shape: {embedding.shape}")
            typer.echo("Note: Full nearest-neighbor search not yet implemented")

    except Exception as e:
        logger.error(f"Encoding failed: {e}")
        raise typer.Exit(1)


@app.command()
def decode(
    glyph: str = typer.Argument(..., help="Glyph string to decode"),
    tape_path: Optional[Path] = typer.Option("tape/v1/tape_index.db", "--tape", help="Path to tape storage"),
):
    """Decode glyph-coded representation to text."""
    from tape import TapeStorage
    import json

    logger.info(f"Decoding glyph: {glyph}")

    try:
        with TapeStorage(str(tape_path)) as storage:
            storage.connect()
            cluster_info = storage.get_cluster_by_glyph(glyph)

            if cluster_info is None:
                typer.echo(f"Glyph '{glyph}' not found in tape")
                raise typer.Exit(1)

            typer.echo(f"\nGlyph: {glyph}")
            typer.echo(f"Cluster ID: {cluster_info['cluster_id']}")
            typer.echo(f"Cluster size: {cluster_info['size']}")
            typer.echo(f"Fractal address: {cluster_info['fractal_address']}")
            typer.echo(f"Coordinates: {cluster_info['coords']}")

            # Show example phrases if available
            metadata = cluster_info.get('metadata', {})
            if isinstance(metadata, str):
                # Parse if it's a string
                try:
                    metadata = eval(metadata)  # Note: Use json.loads in production
                except:
                    metadata = {}

            examples = metadata.get('examples', [])
            if examples:
                typer.echo("\nExample phrases:")
                for ex in examples[:5]:
                    if isinstance(ex, dict):
                        typer.echo(f"  - {ex.get('text', '')}")

    except Exception as e:
        logger.error(f"Decoding failed: {e}")
        raise typer.Exit(1)


@app.command()
def inspect_glyph(
    glyph: str = typer.Argument(..., help="Glyph to inspect"),
    tape_path: Optional[Path] = typer.Option("tape/v1/tape_index.db", "--tape", help="Path to tape storage"),
):
    """Inspect cluster details for a glyph."""
    # Same as decode for now
    decode(glyph, tape_path)


@app.command()
def version():
    """Show FGT version."""
    try:
        from fgt import __version__
        typer.echo(f"Fractal Glyph Tape v{__version__}")
    except:
        typer.echo("Fractal Glyph Tape v0.1.0")


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
