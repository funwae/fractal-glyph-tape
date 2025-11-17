#!/usr/bin/env python3
"""Launch script for Fractal Glyph Memory Service.

Usage:
    python scripts/run_memory_server.py [--host HOST] [--port PORT] [--db-path PATH]

Example:
    python scripts/run_memory_server.py --host 0.0.0.0 --port 8001
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    """Launch the memory server."""
    parser = argparse.ArgumentParser(
        description="Launch Fractal Glyph Memory Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port to bind to (default: 8001)",
    )

    parser.add_argument(
        "--db-path",
        type=str,
        default="memory.db",
        help="Path to SQLite database (default: memory.db)",
    )

    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Log level (default: info)",
    )

    args = parser.parse_args()

    # Import here to avoid loading before args are parsed
    from memory.api import create_memory_api
    from memory.service import create_memory_service
    from memory.store import SQLiteMemoryStore
    from memory.policy import MemoryPolicy, MemoryPolicyConfig

    print(f"🧠 Fractal Glyph Memory Service")
    print(f"   Database: {args.db_path}")
    print(f"   Server: http://{args.host}:{args.port}")
    print(f"   API docs: http://{args.host}:{args.port}/docs")
    print()

    # Create components
    store = SQLiteMemoryStore(args.db_path)
    policy = MemoryPolicy(MemoryPolicyConfig())
    service = create_memory_service(store=store, policy=policy)

    # Create FastAPI app
    app = create_memory_api(service)

    # Run with uvicorn
    import uvicorn

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
