#!/usr/bin/env python3
"""
Run the visualization server for Fractal Glyph Tape.

Usage:
    python scripts/run_viz_server.py --tape tape/v1/tape_index.db --port 8000
"""

import argparse
import sys
from pathlib import Path
from loguru import logger


def main():
    parser = argparse.ArgumentParser(description="Run FGT visualization server")
    parser.add_argument(
        "--tape",
        default="tape/v1/tape_index.db",
        help="Path to tape database"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run server on"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    args = parser.parse_args()

    # Check if tape database exists
    tape_path = Path(args.tape)
    if not tape_path.exists():
        logger.error(f"Tape database not found: {tape_path}")
        logger.info("Please run the build pipeline first:")
        logger.info("  fgt build --config configs/demo.yaml")
        sys.exit(1)

    logger.info(f"Starting visualization server...")
    logger.info(f"Tape database: {tape_path}")
    logger.info(f"Server: http://{args.host}:{args.port}")
    logger.info(f"Interactive map: http://localhost:{args.port}/viz")
    logger.info(f"API docs: http://localhost:{args.port}/docs")

    # Import and run
    from viz import create_app
    import uvicorn

    app = create_app(str(tape_path))

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
