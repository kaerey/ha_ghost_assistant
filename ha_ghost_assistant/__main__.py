"""Module entry point."""
from __future__ import annotations

import asyncio

from ha_ghost_assistant.app import run


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
