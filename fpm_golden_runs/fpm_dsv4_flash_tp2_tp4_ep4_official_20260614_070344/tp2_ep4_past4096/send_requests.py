#!/usr/bin/env python3
"""Send synthetic-token completion workloads to a Dynamo frontend."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import gzip
import json
import math
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR / "common"))
sys.path.insert(0, str(_THIS_DIR.parent / "common"))

from random_prompt_tokens import (
    load_random_prompt_token_config,
    make_prompt_token_ids,
    sample_prompt_token_ids,
)


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


def estimate_token_count(text):
    """Estimate token count from dataset text without binding to a model tokenizer."""
    text = (text or "").strip()
    if not text:
        return 1
    words = len(re.findall(r"\S+", text))
    char_estimate = math.ceil(len(text) / 4)
    word_estimate = math.ceil(words * 1.35)
    return max(1, min(32768, max(char_estimate, word_estimate)))


def bounded_power_law_values(lo, hi, mean, count, *, seed):
    """Return shuffled bounded values with an approximate target mean."""

    if count <= 0:
        return []
    lo = int(lo)
    hi = int(hi)
    mean = float(mean)
    if lo > hi:
        raise ValueError(f"invalid range: {lo}..{hi}")
    if lo == hi or count == 1:
        return [lo] * count
    if not (lo <= mean <= hi):
        raise ValueError(f"mean {mean} must be within range {lo}..{hi}")

    mean_fraction = (mean - lo) / (hi - lo)
    exponent = max(0.001, (1.0 / max(mean_fraction, 1e-9)) - 1.0)
    values = []
    for index in range(count):
        q = index / (count - 1)
        values.append(round(lo + (hi - lo) * (q**exponent)))
    rng = random.Random(seed)
    rng.shuffle(values)
    return values


def clamp_shape_to_model_len(isl, osl, max_model_len):
    """Clamp a request shape so ISL+OSL fits the vLLM model length."""

    isl = max(1, int(isl))
    osl = max(1, int(osl))
    if max_model_len is None or isl + osl <= max_model_len:
        return isl, osl
    overflow = isl + osl - max_model_len
    if osl > overflow:
        return isl, osl - overflow
    return max(1, max_model_len - 1), 1


def synthetic_real_workload_shapes(
    count,
    *,
    seed,
    max_model_len,
    isl_min,
    isl_max,
    isl_mean,
    osl_min,
    osl_max,
    osl_mean,
):
    """Return deterministic large serving-style ISL/OSL request shapes."""

    isls = bounded_power_law_values(isl_min, isl_max, isl_mean, count, seed=seed)
    osls = bounded_power_law_values(osl_min, osl_max, osl_mean, count, seed=(seed or 0) + 17)
    shapes = [clamp_shape_to_model_len(isl, osl, max_model_len) for isl, osl in zip(isls, osls, strict=True)]
    return shapes


def fallback_real_workload_shapes(count, *, seed, max_model_len):
    """Return deterministic large serving-style ISL/OSL pairs when dataset loading fails."""

    return synthetic_real_workload_shapes(
        count,
        seed=seed,
        max_model_len=max_model_len,
        isl_min=100,
        isl_max=16384,
        isl_mean=4096,
        osl_min=100,
        osl_max=4096,
        osl_mean=1024,
    )


def scale_real_workload_shapes(shapes, *, seed, max_model_len, isl_min, isl_max, isl_mean, osl_min, osl_max, osl_mean):
    """Map dataset request ordering onto a configured large-shape distribution."""

    if not shapes:
        return []
    count = len(shapes)
    target_isls = bounded_power_law_values(isl_min, isl_max, isl_mean, count, seed=seed)
    target_osls = bounded_power_law_values(osl_min, osl_max, osl_mean, count, seed=(seed or 0) + 17)
    ordering = sorted(range(count), key=lambda i: (shapes[i][0], shapes[i][1]))
    scaled = [None] * count
    for rank, index in enumerate(ordering):
        scaled[index] = clamp_shape_to_model_len(target_isls[rank], target_osls[rank], max_model_len)
    return scaled


def load_openassistant_shapes(dataset_name, *, count, seed, max_model_len, max_rows):
    """Sample ISL/OSL pairs from OpenAssistant-style prompt/assistant rows."""
    shapes = load_openassistant_shapes_with_datasets(
        dataset_name,
        count=count,
        seed=seed,
        max_model_len=max_model_len,
        max_rows=max_rows,
    )
    if shapes:
        return shapes
    return load_openassistant_shapes_with_hub_jsonl(
        dataset_name,
        count=count,
        seed=seed,
        max_model_len=max_model_len,
        max_rows=max_rows,
    )


def _append_openassistant_shape(row, prompts_by_id, shapes, max_model_len):
    role = str(row.get("role", "")).lower()
    text = str(row.get("text", "") or "")
    message_id = row.get("message_id")
    parent_id = row.get("parent_id")
    if role in {"prompter", "user"} and message_id:
        prompts_by_id[message_id] = text
        return
    if role != "assistant" or parent_id not in prompts_by_id:
        return

    isl = estimate_token_count(prompts_by_id[parent_id])
    osl = estimate_token_count(text)
    shapes.append(clamp_shape_to_model_len(isl, osl, max_model_len))


def load_openassistant_shapes_with_datasets(dataset_name, *, count, seed, max_model_len, max_rows):
    """Sample OASST shapes through the optional datasets package."""
    try:
        from datasets import load_dataset
    except Exception as exc:
        print(f"real_workload_dataset_unavailable reason={type(exc).__name__}", flush=True)
        return []

    try:
        dataset = load_dataset(dataset_name, split="train", streaming=True)
    except Exception as exc:
        print(f"real_workload_dataset_load_failed dataset={dataset_name!r} error={exc!r}", flush=True)
        return []

    prompts_by_id = {}
    shapes = []
    rng = random.Random(seed)
    for idx, row in enumerate(dataset):
        if idx >= max_rows or len(shapes) >= max(count * 4, count):
            break
        if row.get("lang") not in (None, "", "en"):
            continue
        _append_openassistant_shape(row, prompts_by_id, shapes, max_model_len)

    if not shapes:
        return []
    rng.shuffle(shapes)
    while len(shapes) < count:
        shapes.extend(shapes[: count - len(shapes)])
    return shapes[:count]


def _flatten_tree_messages(node):
    if not isinstance(node, dict):
        return
    yield node
    for reply in node.get("replies") or []:
        yield from _flatten_tree_messages(reply)


def _oasst_rows_from_jsonl(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if "prompt" in obj:
                yield from _flatten_tree_messages(obj["prompt"])
            else:
                yield obj


def load_openassistant_shapes_with_hub_jsonl(dataset_name, *, count, seed, max_model_len, max_rows):
    """Sample OASST shapes from HF Hub JSONL exports without datasets."""
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except Exception as exc:
        print(f"real_workload_hub_unavailable reason={type(exc).__name__}", flush=True)
        return []

    try:
        files = list_repo_files(dataset_name, repo_type="dataset")
        candidates = [name for name in files if name.endswith(".jsonl.gz") and ("messages" in name or "trees" in name)]
        if not candidates:
            print(f"real_workload_hub_no_jsonl dataset={dataset_name!r}", flush=True)
            return []
        preferred = sorted(candidates, key=lambda name: ("ready" not in name, "messages" not in name, name))[0]
        path = hf_hub_download(dataset_name, preferred, repo_type="dataset")
    except Exception as exc:
        print(f"real_workload_hub_load_failed dataset={dataset_name!r} error={exc!r}", flush=True)
        return []

    prompts_by_id = {}
    shapes = []
    rng = random.Random(seed)
    try:
        for idx, row in enumerate(_oasst_rows_from_jsonl(path)):
            if idx >= max_rows or len(shapes) >= max(count * 4, count):
                break
            if row.get("lang") not in (None, "", "en"):
                continue
            _append_openassistant_shape(row, prompts_by_id, shapes, max_model_len)
    except Exception as exc:
        print(f"real_workload_hub_parse_failed path={path!r} error={exc!r}", flush=True)
        return []

    if not shapes:
        return []
    rng.shuffle(shapes)
    while len(shapes) < count:
        shapes.extend(shapes[: count - len(shapes)])
    return shapes[:count]


def real_workload_values(args):
    """Return realistic ISL/OSL pairs for the requested number of requests."""
    max_model_len = getattr(args, "max_model_len", None)
    if max_model_len is not None:
        max_model_len = max(2, int(max_model_len))
    count = int(args.requests)
    dataset_name = getattr(args, "real_workload_dataset", "OpenAssistant/oasst1")
    max_rows = int(getattr(args, "real_workload_max_rows", 5000))
    shape_source = getattr(args, "real_workload_shape_source", "scaled_dataset")
    range_kwargs = {
        "isl_min": int(getattr(args, "real_workload_isl_min", 100)),
        "isl_max": int(getattr(args, "real_workload_isl_max", 16384)),
        "isl_mean": float(getattr(args, "real_workload_isl_mean", 4096)),
        "osl_min": int(getattr(args, "real_workload_osl_min", 100)),
        "osl_max": int(getattr(args, "real_workload_osl_max", 4096)),
        "osl_mean": float(getattr(args, "real_workload_osl_mean", 1024)),
    }
    if shape_source == "synthetic":
        shapes = synthetic_real_workload_shapes(
            count,
            seed=args.prompt_token_seed,
            max_model_len=max_model_len,
            **range_kwargs,
        )
        source = "synthetic_large_shape_distribution"
        print(f"real_workload_shapes source={source!r} count={len(shapes)}", flush=True)
        return shapes, source

    shapes = load_openassistant_shapes(
        dataset_name,
        count=count,
        seed=args.prompt_token_seed,
        max_model_len=max_model_len,
        max_rows=max_rows,
    )
    source = f"{dataset_name}:scaled_large_shape_distribution"
    shapes = scale_real_workload_shapes(
        shapes,
        seed=args.prompt_token_seed,
        max_model_len=max_model_len,
        **range_kwargs,
    )
    if not shapes:
        shapes = synthetic_real_workload_shapes(
            count,
            seed=args.prompt_token_seed,
            max_model_len=max_model_len,
            **range_kwargs,
        )
        source = "synthetic_large_shape_distribution"
    print(f"real_workload_shapes source={source!r} count={len(shapes)}", flush=True)
    return shapes, source


def make_token_ids(args, target_tokens, request_index):
    token_pool = getattr(args, "prompt_token_pool", None)
    if args.prompt_token_seed is None:
        rng = args.prompt_rng
    else:
        rng = random.Random(int(args.prompt_token_seed) + int(request_index))
    if token_pool:
        return [token_pool[rng.randrange(len(token_pool))] for _ in range(int(target_tokens))]
    if args.prompt_token_seed is None:
        return sample_prompt_token_ids(
            rng,
            int(target_tokens),
            args.prompt_token_config,
        )
    return make_prompt_token_ids(
        prompt_token_seed=args.prompt_token_seed,
        token_count=int(target_tokens),
        request_index=request_index,
        token_config=args.prompt_token_config,
    )


def is_printable_ascii_token_text(text):
    """Return whether one decoded token is safe for vLLM HTTP prompt handling."""

    if not text or not text.strip():
        return False
    return all(32 <= ord(ch) <= 126 for ch in text)


def decode_one_token(tokenizer, token_id):
    try:
        return tokenizer.decode(
            [token_id],
            clean_up_tokenization_spaces=False,
            skip_special_tokens=False,
        )
    except TypeError:
        return tokenizer.decode([token_id])


def resolve_cached_hf_file(model, filename):
    try:
        from huggingface_hub import hf_hub_download
    except Exception:
        return None
    try:
        return hf_hub_download(repo_id=model, filename=filename, local_files_only=True)
    except Exception:
        return None


def iter_printable_ascii_token_ids(token_config, decode_token):
    candidates = []
    for token_id in range(token_config.vocab_size):
        if token_id in token_config.excluded_token_ids:
            continue
        try:
            text = decode_token(token_id)
        except Exception:
            continue
        if is_printable_ascii_token_text(text):
            candidates.append(token_id)
    return candidates


def load_safe_ascii_prompt_token_ids_from_tokenizer_json(model, token_config):
    tokenizer_path = resolve_cached_hf_file(model, "tokenizer.json")
    if not tokenizer_path:
        return []
    try:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(tokenizer_path)
    except Exception as exc:
        print(
            f"safe_ascii_tokenizer_json_load_failed model={model!r} error={exc!r}",
            flush=True,
        )
        return []
    return iter_printable_ascii_token_ids(
        token_config,
        lambda token_id: tokenizer.decode([token_id], skip_special_tokens=False),
    )


def load_safe_ascii_prompt_token_ids(model, token_config):
    load_error = None
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        print(f"safe_ascii_prompt_tokens_unavailable reason={type(exc).__name__}", flush=True)
        load_error = exc
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        except Exception as exc:
            print(f"safe_ascii_prompt_tokens_load_failed model={model!r} error={exc!r}", flush=True)
            load_error = exc
        else:
            candidates = iter_printable_ascii_token_ids(
                token_config,
                lambda token_id: decode_one_token(tokenizer, token_id),
            )
            if not candidates:
                print("safe_ascii_prompt_tokens_empty", flush=True)
            return candidates

    candidates = load_safe_ascii_prompt_token_ids_from_tokenizer_json(model, token_config)
    if candidates:
        if load_error is not None:
            print(
                f"safe_ascii_prompt_tokens_loaded_from=tokenizer_json after_error={type(load_error).__name__}",
                flush=True,
            )
        return candidates
    if not candidates:
        print("safe_ascii_prompt_tokens_empty", flush=True)
    return candidates


def build_specs(args):
    if args.endpoint != "completions":
        raise ValueError("random token-id prompts require endpoint=completions")

    real_workload = bool(getattr(args, "real_workload", False))
    real_source = ""
    if real_workload:
        shapes, real_source = real_workload_values(args)
        isls = [shape[0] for shape in shapes]
        osls = [shape[1] for shape in shapes]
    elif args.vary_isl_osl:
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
                    "shape_source",
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
                    getattr(args, "prompt_token_mode", "random_vocab_excluding_special"),
                    real_source if real_workload else "explicit_or_logspace_shape_grid",
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
    parser.add_argument("--prompt-token-seed", type=int, default=None)
    parser.add_argument(
        "--prompt-token-mode",
        choices=["random_vocab_excluding_special", "safe_ascii"],
        default="random_vocab_excluding_special",
    )
    parser.add_argument("--vary-isl-osl", action="store_true")
    parser.add_argument("--real-workload", action="store_true")
    parser.add_argument("--real-workload-dataset", default="OpenAssistant/oasst1")
    parser.add_argument("--real-workload-max-rows", type=int, default=5000)
    parser.add_argument(
        "--real-workload-shape-source",
        choices=["scaled_dataset", "synthetic"],
        default="scaled_dataset",
    )
    parser.add_argument("--real-workload-isl-min", type=int, default=100)
    parser.add_argument("--real-workload-isl-max", type=int, default=16384)
    parser.add_argument("--real-workload-isl-mean", type=float, default=4096)
    parser.add_argument("--real-workload-osl-min", type=int, default=100)
    parser.add_argument("--real-workload-osl-max", type=int, default=4096)
    parser.add_argument("--real-workload-osl-mean", type=float, default=1024)
    parser.add_argument("--endpoint", choices=["completions"], default="completions")
    parser.add_argument("--isl-min", type=int, default=1)
    parser.add_argument("--isl-max", type=int, default=4096)
    parser.add_argument("--osl-min", type=int, default=1)
    parser.add_argument("--osl-max", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=None)
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
    args.prompt_rng = random.Random(args.prompt_token_seed)
    args.prompt_token_config = load_random_prompt_token_config(
        args.model,
        allow_transformers_fallback=True,
    )
    args.prompt_token_pool = []
    if args.prompt_token_mode == "safe_ascii":
        args.prompt_token_pool = load_safe_ascii_prompt_token_ids(args.model, args.prompt_token_config)
        if args.prompt_token_pool:
            print(
                f"prompt_token_mode=safe_ascii candidate_count={len(args.prompt_token_pool)}",
                flush=True,
            )
        else:
            args.prompt_token_mode = "random_vocab_excluding_special"
            print("prompt_token_mode_fallback=random_vocab_excluding_special", flush=True)
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
        f"completed={ok}/{args.requests} failed={failed} elapsed_seconds={elapsed:.3f}",
        flush=True,
    )
    return 0 if ok + failed == args.requests and failed <= args.allow_failures else 1


if __name__ == "__main__":
    sys.exit(main())
