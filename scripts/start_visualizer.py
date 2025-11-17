#!/usr/bin/env python3
"""Start the FastAPI backend server for the visualizer."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Start the Fractal Glyph Tape visualizer API server"
    )

    parser.add_argument(
        "--tape-dir",
        type=str,
        default="tape/v1",
        help="Path to tape directory (default: tape/v1)"
    )

    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )

    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )

    args = parser.parse_args()

    # Validate tape directory
    tape_path = Path(args.tape_dir)
    if not tape_path.exists():
        print(f"Error: Tape directory not found: {tape_path}", file=sys.stderr)
        print("\nPlease build a tape first using:", file=sys.stderr)
        print("  python -m src.fgt.cli build --config configs/demo.yaml", file=sys.stderr)
        sys.exit(1)

    clusters_dir = tape_path / "clusters"
    if not clusters_dir.exists():
        print(f"Error: Clusters directory not found: {clusters_dir}", file=sys.stderr)
        sys.exit(1)

    # Check if layout exists
    layout_file = clusters_dir / "layout.npy"
    if not layout_file.exists():
        print(f"Warning: Layout file not found: {layout_file}", file=sys.stderr)
        print("The visualizer will not be able to display the map.", file=sys.stderr)
        print("\nGenerate layout using:", file=sys.stderr)
        print(f"  python -m src.viz.layout {clusters_dir}", file=sys.stderr)
        print()

    print("=" * 60)
    print("Fractal Glyph Tape Visualizer API Server")
    print("=" * 60)
    print(f"Tape directory: {tape_path.resolve()}")
    print(f"Server: http://{args.host}:{args.port}")
    print(f"Reload: {'enabled' if args.reload else 'disabled'}")
    print("=" * 60)
    print()
    print("Press Ctrl+C to stop the server")
    print()

    # Import here to set tape_dir via environment
    import os
    os.environ["TAPE_DIR"] = str(tape_path)

    # Start uvicorn
    try:
        import uvicorn
        uvicorn.run(
            "src.viz.api:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )
    except KeyboardInterrupt:
        print("\nServer stopped")
    except ImportError:
        print("Error: uvicorn not installed", file=sys.stderr)
        print("Install with: pip install uvicorn", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
