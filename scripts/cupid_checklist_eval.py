"""Evaluate GPT-4o Mini on CUPID checklist compliance."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

from openai import OpenAI

from cupid_evaluation import (
    DEFAULT_MODEL,
    EvaluationConfig,
    evaluate_dataset,
    load_json_dataset,
)

LOGGER = logging.getLogger(__name__)


def _load_dataset(path: str) -> List[Dict[str, Any]]:
    try:
        return load_json_dataset(path)
    except FileNotFoundError as exc:  # pragma: no cover - CLI validation
        raise SystemExit(f"Input file not found: {path}") from exc
    except json.JSONDecodeError as exc:  # pragma: no cover - CLI validation
        raise SystemExit(f"Failed to parse JSON input: {exc}") from exc


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/CUPID/test.json", help="Path to the CUPID JSON dataset.")
    parser.add_argument("--output", help="Optional path to store per-sample predictions as JSON Lines.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model name (default: %(default)s).")
    parser.add_argument("--api-key", dest="api_key", help="OpenAI API key (overrides OPENAI_API_KEY env var).")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum retries for API calls.")
    parser.add_argument("--retry-sleep", type=float, default=1.0, help="Seconds to sleep between retries.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: %(default)s).")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    dataset = _load_dataset(args.input)

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:  # pragma: no cover - CLI validation
        raise SystemExit("An OpenAI API key must be provided via --api-key or OPENAI_API_KEY.")

    client = OpenAI(api_key=api_key)
    config = EvaluationConfig(
        model=args.model,
        max_retries=args.max_retries,
        retry_sleep=args.retry_sleep,
    )

    results = evaluate_dataset(dataset, client=client, config=config)

    LOGGER.info(
        "Checklist accuracy: %.4f (%s/%s)",
        results["accuracy"],
        results["correct_items"],
        results["total_items"],
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            for sample in results["predictions"]:
                fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
        LOGGER.info("Saved predictions to %s", args.output)

    print(json.dumps(
        {
            "accuracy": results["accuracy"],
            "correct_items": results["correct_items"],
            "total_items": results["total_items"],
        },
        ensure_ascii=False,
    ))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
