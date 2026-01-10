"""Module entry point."""
from __future__ import annotations

import argparse
import asyncio
import os

from ha_ghost_assistant.app import run


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HA Ghost Assistant")
    parser.add_argument(
        "--host",
        default=os.getenv("HA_GHOST_ASSISTANT_HOST", "0.0.0.0"),
        help="Host interface to bind Wyoming server (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("HA_GHOST_ASSISTANT_PORT", "10700")),
        help="Port to bind Wyoming server (default: 10700)",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    asyncio.run(run(args.host, args.port))


if __name__ == "__main__":
    main()
