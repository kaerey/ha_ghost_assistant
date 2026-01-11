#!/usr/bin/env python3
"""Simple Wyoming protocol smoke test for describe and ping."""
from __future__ import annotations

import argparse
import json
import socket
import sys


def _read_line(sock: socket.socket) -> dict[str, object]:
    buffer = b""
    while b"\n" not in buffer:
        chunk = sock.recv(4096)
        if not chunk:
            raise RuntimeError("Connection closed while waiting for response")
        buffer += chunk
    line, _rest = buffer.split(b"\n", 1)
    return json.loads(line.decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=12345)
    args = parser.parse_args()

    with socket.create_connection((args.host, args.port), timeout=5) as sock:
        describe = {"type": "describe", "version": "1.7.2"}
        sock.sendall((json.dumps(describe) + "\n").encode("utf-8"))
        first = _read_line(sock)
        if first.get("type") != "info":
            raise AssertionError(f"Expected info response, got {first!r}")
        ping = {"type": "ping", "data": {"text": "test"}}
        sock.sendall((json.dumps(ping) + "\n").encode("utf-8"))
        second = _read_line(sock)
        if second.get("type") != "pong":
            raise AssertionError(f"Expected pong response, got {second!r}")

    print("Wyoming describe/ping test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
