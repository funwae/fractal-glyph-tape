#!/usr/bin/env python3
"""Debug utility to inspect memory store contents.

Usage:
    python scripts/debug_memory_dump.py [--db-path PATH] [--actor-id ID] [--region REGION]

Example:
    python scripts/debug_memory_dump.py --actor-id hayden
    python scripts/debug_memory_dump.py --actor-id hayden --region "agent:hayden"
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    """Dump memory store contents."""
    parser = argparse.ArgumentParser(
        description="Inspect Fractal Glyph Memory Store",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--db-path",
        type=str,
        default="memory.db",
        help="Path to SQLite database (default: memory.db)",
    )

    parser.add_argument(
        "--actor-id",
        type=str,
        help="Filter by actor ID",
    )

    parser.add_argument(
        "--region",
        type=str,
        help="Filter by region",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of records to show (default: 10)",
    )

    parser.add_argument(
        "--show-spans",
        action="store_true",
        help="Show detailed span information",
    )

    args = parser.parse_args()

    from memory.store import SQLiteMemoryStore
    from memory.addresses import address_to_str

    # Check if database exists
    if not Path(args.db_path).exists():
        print(f"❌ Database not found: {args.db_path}")
        sys.exit(1)

    store = SQLiteMemoryStore(args.db_path)

    if not args.actor_id:
        print("❌ Please specify --actor-id")
        sys.exit(1)

    # List regions if no specific region requested
    if not args.region:
        print(f"🔍 Regions for actor '{args.actor_id}':\n")
        regions = store.list_regions(args.actor_id)

        if not regions:
            print("   (no regions found)")
        else:
            for region in regions:
                stats = store.get_region_stats(args.actor_id, region)
                print(f"   📁 {region}")
                print(f"      Records: {stats['record_count']}")
                print(f"      Spans: {stats['span_count']}")
                if stats['first_timestamp']:
                    print(f"      First: {stats['first_timestamp'].isoformat()}")
                if stats['last_timestamp']:
                    print(f"      Last: {stats['last_timestamp'].isoformat()}")
                print()

        print(f"\n💡 Use --region REGION to see detailed records")
        return

    # Show detailed records for region
    print(f"📝 Records for actor '{args.actor_id}' in region '{args.region}':\n")

    records = list(store.get_records_for_actor(args.actor_id, args.region, limit=args.limit))

    if not records:
        print("   (no records found)")
        return

    for i, record in enumerate(records, 1):
        print(f"   [{i}] Record ID: {record.id}")
        print(f"       Created: {record.created_at.isoformat()}")
        print(f"       Source: {record.source}")
        print(f"       Tags: {', '.join(record.tags) if record.tags else '(none)'}")
        print(f"       Text: {record.raw_text[:100]}{'...' if len(record.raw_text) > 100 else ''}")

        if args.show_spans:
            print(f"       Spans: {len(record.spans)}")
            for j, span in enumerate(record.spans, 1):
                print(f"         [{j}] Address: {address_to_str(span.address)}")
                print(f"             Glyphs: {', '.join(span.glyph_ids)}")
                print(f"             Text: {span.text[:80]}{'...' if len(span.text) > 80 else ''}")
                print(f"             Language: {span.language}")
        else:
            print(f"       Spans: {len(record.spans)} (use --show-spans for details)")

        print()


if __name__ == "__main__":
    main()
