#!/usr/bin/env python3
"""
Start the Fractal Glyph Memory System API server

Usage:
    python scripts/start_memory_api.py [--port 8001] [--reload]
"""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import uvicorn
import argparse


def main():
    parser = argparse.ArgumentParser(description="Start FGMS API server")
    parser.add_argument("--port", type=int, default=8001, help="Port to run on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    print(f"Starting Fractal Glyph Memory System API on {args.host}:{args.port}")
    print(f"API documentation: http://localhost:{args.port}/docs")
    print(f"Memory Console: http://localhost:3000/memory-console")
    print("\nPress Ctrl+C to stop")

    uvicorn.run(
        "memory_system.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()
