"""Command-line interface for Fractal Glyph Tape.

Usage:
    fgt build --config configs/demo.yaml
    fgt encode "Can you send me that file?"
    fgt decode "谷阜"
    fgt inspect-glyph 谷阜
"""

import typer
from pathlib import Path
from typing import Optional

app = typer.Typer(help="Fractal Glyph Tape - Semantic compression and phrase memory")


@app.command()
def build(
    config: Path = typer.Option(..., "--config", help="Path to config YAML file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Run the full FGT build pipeline."""
    typer.echo(f"Building FGT from config: {config}")
    typer.echo("This feature is not yet implemented.")
    typer.echo("See claude.md for implementation instructions.")


@app.command()
def encode(
    text: str = typer.Argument(..., help="Text to encode as glyph representation"),
    tape_path: Optional[Path] = typer.Option(None, "--tape", help="Path to tape storage"),
):
    """Encode text to glyph-coded representation."""
    typer.echo(f"Encoding: {text}")
    typer.echo("This feature is not yet implemented.")
    typer.echo("See claude.md for implementation instructions.")


@app.command()
def decode(
    glyph: str = typer.Argument(..., help="Glyph string to decode"),
    tape_path: Optional[Path] = typer.Option(None, "--tape", help="Path to tape storage"),
):
    """Decode glyph-coded representation to text."""
    typer.echo(f"Decoding glyph: {glyph}")
    typer.echo("This feature is not yet implemented.")
    typer.echo("See claude.md for implementation instructions.")


@app.command()
def inspect_glyph(
    glyph: str = typer.Argument(..., help="Glyph to inspect"),
    tape_path: Optional[Path] = typer.Option(None, "--tape", help="Path to tape storage"),
):
    """Inspect cluster details for a glyph."""
    typer.echo(f"Inspecting glyph: {glyph}")
    typer.echo("This feature is not yet implemented.")
    typer.echo("See claude.md for implementation instructions.")


@app.command()
def version():
    """Show FGT version."""
    from fgt import __version__
    typer.echo(f"Fractal Glyph Tape v{__version__}")


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
