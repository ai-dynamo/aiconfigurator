#!/usr/bin/env python3
"""Wait for a local HTTP endpoint to become ready."""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--contains", default=None)
    args = parser.parse_args()

    deadline = time.monotonic() + args.timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(args.url, timeout=5.0) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                if 200 <= resp.status < 300 and (args.contains is None or args.contains in body):
                    print(body[:1000])
                    return 0
                last_error = f"status={resp.status}, missing={args.contains!r}"
        except Exception as exc:
            last_error = repr(exc)
        time.sleep(2.0)
    print(f"timed out waiting for {args.url}: {last_error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
