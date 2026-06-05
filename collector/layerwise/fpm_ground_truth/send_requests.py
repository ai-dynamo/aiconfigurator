#!/usr/bin/env python3
"""Send deterministic random-token completion workloads to a Dynamo frontend."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR / "common"))

from random_prompt_tokens import load_random_prompt_token_config, make_prompt_token_ids


def post_json(url, payload, timeout):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read()


def describe_http_error(exc):
    try:
        body = exc.read().decode("utf-8", errors="replace").strip()
    except Exception:
        body = ""
    if len(body) > 500:
        body = body[:500] + "..."
    return f"HTTP {exc.code}: {body}" if body else f"HTTP {exc.code}"


def parse_values(text):
    if not text:
        return []
    values = []
    for part in text.split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    if not values:
        raise ValueError("empty explicit value list")
    return values


def logspace_ints(lo, hi, count):
    if count <= 0:
        return []
    if count == 1 or lo == hi:
        return [lo] * count
    lo_log = math.log(lo)
    hi_log = math.log(hi)
    values = []
    for i in range(count):
        ratio = i / (count - 1)
        value = round(math.exp(lo_log + ratio * (hi_log - lo_log)))
        values.append(max(lo, min(hi, value)))
    values[0] = lo
    values[-1] = hi
    return values


def target_values(explicit, lo, hi, count):
    values = parse_values(explicit)
    if values:
        return values
    return logspace_ints(lo, hi, count)


def make_token_ids(args, target_tokens, request_index):
    return make_prompt_token_ids(
        prompt_token_seed=args.prompt_token_seed,
        token_count=int(target_tokens),
        request_index=request_index,
        token_config=args.prompt_token_config,
    )


def build_specs(args):
    if args.endpoint != "completions":
        raise ValueError("random token-id prompts require endpoint=completions")

    if args.vary_isl_osl:
        isls = target_values(args.isl_values, args.isl_min, args.isl_max, args.requests)
        osls = target_values(args.osl_values, args.osl_min, args.osl_max, args.requests)
    else:
        isls = [args.isl_min]
        osls = [args.max_tokens]

    specs = []
    for i in range(args.requests):
        request_index = args.request_index_offset + i
        target_isl = isls[i % len(isls)]
        target_osl = osls[i % len(osls)]
        prompt = make_token_ids(args, target_isl, request_index)
        prompt_tokens = len(prompt)

        specs.append(
            {
                "index": request_index,
                "prompt": prompt,
                "target_isl": int(target_isl),
                "target_osl": int(target_osl),
                "prompt_tokens": int(prompt_tokens),
            }
        )

    output_path = Path(args.workload_output)
    write_header = not args.append_workload or not output_path.exists() or output_path.stat().st_size == 0
    mode = "a" if args.append_workload else "w"
    with output_path.open(mode, newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                [
                    "workload_label",
                    "request_index",
                    "endpoint",
                    "target_isl",
                    "target_osl",
                    "prompt_tokens",
                    "max_tokens",
                    "prompt_token_mode",
                ]
            )
        for spec in specs:
            writer.writerow(
                [
                    args.workload_label,
                    spec["index"],
                    args.endpoint,
                    spec["target_isl"],
                    spec["target_osl"],
                    spec["prompt_tokens"],
                    spec["target_osl"],
                    "random_vocab_excluding_special",
                ]
            )

    return specs


def send_one(spec, args):
    payload = {
        "model": args.model,
        "max_tokens": spec["target_osl"],
        "temperature": 0,
        "stream": False,
    }
    if args.ignore_eos:
        payload["ignore_eos"] = True

    if args.endpoint == "completions":
        payload["prompt"] = spec["prompt"]
        url = f"{args.url.rstrip('/')}/v1/completions"
    else:
        raise ValueError(f"unsupported endpoint: {args.endpoint}")

    transient_statuses = {408, 409, 429, 500, 502, 503, 504}
    start = time.monotonic()
    for attempt in range(args.retries + 1):
        try:
            status = post_json(url, payload, args.timeout)[0]
            elapsed = time.monotonic() - start
            return spec, status, elapsed
        except urllib.error.HTTPError as exc:
            if exc.code not in transient_statuses or attempt >= args.retries:
                raise RuntimeError(describe_http_error(exc)) from exc
            sleep_seconds = args.retry_backoff * (attempt + 1)
            print(
                "request_retry "
                f"index={spec['index']} "
                f"status={exc.code} "
                f"attempt={attempt + 1}/{args.retries} "
                f"sleep_seconds={sleep_seconds:.1f}",
                flush=True,
            )
            time.sleep(sleep_seconds)
        except (TimeoutError, urllib.error.URLError) as exc:
            if attempt >= args.retries:
                raise
            sleep_seconds = args.retry_backoff * (attempt + 1)
            print(
                "request_retry "
                f"index={spec['index']} "
                f"error={exc!r} "
                f"attempt={attempt + 1}/{args.retries} "
                f"sleep_seconds={sleep_seconds:.1f}",
                flush=True,
            )
            time.sleep(sleep_seconds)

    raise RuntimeError("unreachable retry loop exit")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--requests", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--prompt-token-seed", type=int, default=0)
    parser.add_argument("--vary-isl-osl", action="store_true")
    parser.add_argument("--endpoint", choices=["completions"], default="completions")
    parser.add_argument("--isl-min", type=int, default=1)
    parser.add_argument("--isl-max", type=int, default=4096)
    parser.add_argument("--osl-min", type=int, default=1)
    parser.add_argument("--osl-max", type=int, default=1024)
    parser.add_argument("--isl-values", default="")
    parser.add_argument("--osl-values", default="")
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--workload-output", required=True)
    parser.add_argument("--workload-label", default="measured")
    parser.add_argument("--append-workload", action="store_true")
    parser.add_argument("--request-index-offset", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-backoff", type=float, default=2.0)
    parser.add_argument("--allow-failures", type=int, default=0)
    args = parser.parse_args()
    args.prompt_token_config = load_random_prompt_token_config(
        args.model,
        allow_transformers_fallback=True,
    )
    print(
        "prompt_token_config "
        f"vocab_size={args.prompt_token_config.vocab_size} "
        f"excluded_token_count={len(args.prompt_token_config.excluded_token_ids)}",
        flush=True,
    )

    specs = build_specs(args)
    print(f"wrote_workload={args.workload_output}", flush=True)

    start = time.monotonic()
    ok = 0
    failed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(send_one, spec, args) for spec in specs]
        for fut in concurrent.futures.as_completed(futures):
            try:
                spec, status, elapsed = fut.result()
                if 200 <= status < 300:
                    ok += 1
                    print(
                        "request_ok "
                        f"index={spec['index']} "
                        f"status={status} "
                        f"target_isl={spec['target_isl']} "
                        f"target_osl={spec['target_osl']} "
                        f"prompt_tokens={spec['prompt_tokens']} "
                        f"elapsed_seconds={elapsed:.3f}",
                        flush=True,
                    )
                else:
                    print(
                        f"request_bad_status index={spec['index']} status={status}",
                        flush=True,
                    )
            except Exception as exc:
                failed += 1
                print(f"request_failed error={exc!r}", flush=True)

    elapsed = time.monotonic() - start
    print(
        f"completed={ok}/{args.requests} "
        f"failed={failed} "
        f"elapsed_seconds={elapsed:.3f}",
        flush=True,
    )
    return 0 if ok + failed == args.requests and failed <= args.allow_failures else 1


if __name__ == "__main__":
    sys.exit(main())
