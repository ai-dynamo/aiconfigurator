"""Utilities for deterministic synthetic token-id prompts."""

from __future__ import annotations

import json
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SPECIAL_TOKEN_ID_KEYS = (
    "bos_token_id",
    "eos_token_id",
    "pad_token_id",
    "unk_token_id",
    "sep_token_id",
    "cls_token_id",
    "mask_token_id",
    "decoder_start_token_id",
)


@dataclass(frozen=True)
class RandomPromptTokenConfig:
    vocab_size: int
    excluded_token_ids: frozenset[int]


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        return {}
    return payload


def read_hf_json_if_available(model: str, filename: str) -> dict[str, Any]:
    try:
        from huggingface_hub import hf_hub_download
    except Exception:
        return {}

    try:
        path = hf_hub_download(repo_id=model, filename=filename, local_files_only=True)
    except Exception:
        return {}
    return read_json_if_exists(Path(path))


def iter_token_ids(value: Any) -> Iterable[int]:
    if value is None:
        return
    if isinstance(value, bool):
        return
    if isinstance(value, int):
        yield value
        return
    if isinstance(value, str):
        try:
            yield int(value)
        except ValueError:
            return
        return
    if isinstance(value, Iterable) and not isinstance(value, dict):
        for item in value:
            yield from iter_token_ids(item)


def load_random_prompt_token_config(
    model: str,
    *,
    allow_transformers_fallback: bool = False,
) -> RandomPromptTokenConfig:
    model_path = Path(model)
    config = read_json_if_exists(model_path / "config.json")
    generation_config = read_json_if_exists(model_path / "generation_config.json")
    tokenizer_config = read_json_if_exists(model_path / "tokenizer_config.json")

    if allow_transformers_fallback and not config and not tokenizer_config:
        fallback_error: Exception | None = None
        try:
            from transformers import AutoConfig, AutoTokenizer

            try:
                hf_config = AutoConfig.from_pretrained(model, trust_remote_code=True)
                config = dict(getattr(hf_config, "to_dict", lambda: {})())
            except Exception as exc:
                config = {}
                fallback_error = exc
            try:
                tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
                tokenizer_config = {"vocab_size": getattr(tokenizer, "vocab_size", None)}
                tokenizer_config["special_token_ids"] = list(getattr(tokenizer, "all_special_ids", []))
            except Exception as exc:
                fallback_error = exc
        except Exception as exc:
            fallback_error = exc

        if not config:
            config = read_hf_json_if_available(model, "config.json")
        if not generation_config:
            generation_config = read_hf_json_if_available(model, "generation_config.json")
        if not tokenizer_config:
            tokenizer_config = read_hf_json_if_available(model, "tokenizer_config.json")
        special_tokens_map = read_hf_json_if_available(model, "special_tokens_map.json")

        if not config and not tokenizer_config:
            raise ValueError(
                f"could not determine random prompt token vocabulary for {model!r}"
            ) from fallback_error
    else:
        special_tokens_map = read_json_if_exists(model_path / "special_tokens_map.json")

    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        text_config = {}

    raw_vocab_size = (
        config.get("vocab_size")
        or text_config.get("vocab_size")
        or tokenizer_config.get("vocab_size")
    )
    if raw_vocab_size is None:
        raise ValueError(f"could not determine vocab_size for {model!r}")
    vocab_size = int(raw_vocab_size)
    if vocab_size <= 0:
        raise ValueError(f"invalid vocab_size for random prompt IDs: {vocab_size}")

    excluded: set[int] = set()
    for payload in (config, text_config, generation_config, tokenizer_config, special_tokens_map):
        for key in SPECIAL_TOKEN_ID_KEYS:
            excluded.update(iter_token_ids(payload.get(key)))
    excluded.update(iter_token_ids(tokenizer_config.get("special_token_ids")))

    added_tokens = tokenizer_config.get("added_tokens_decoder", {})
    if isinstance(added_tokens, dict):
        for raw_token_id, token_payload in added_tokens.items():
            if isinstance(token_payload, dict) and token_payload.get("special") is True:
                excluded.update(iter_token_ids(raw_token_id))

    excluded = {token_id for token_id in excluded if 0 <= token_id < vocab_size}
    if len(excluded) >= vocab_size:
        raise ValueError("special-token exclusion covers the whole vocabulary")
    return RandomPromptTokenConfig(vocab_size=vocab_size, excluded_token_ids=frozenset(excluded))


def sample_prompt_token_ids(
    rng: Any,
    token_count: int,
    token_config: RandomPromptTokenConfig,
) -> list[int]:
    tokens: list[int] = []
    while len(tokens) < token_count:
        token_id = rng.randrange(token_config.vocab_size)
        if token_id not in token_config.excluded_token_ids:
            tokens.append(token_id)
    return tokens


def make_prompt_token_ids(
    *,
    prompt_token_seed: int,
    token_count: int,
    request_index: int,
    token_config: RandomPromptTokenConfig,
) -> list[int]:
    rng = random.Random(int(prompt_token_seed) + int(request_index))
    return sample_prompt_token_ids(rng, int(token_count), token_config)
