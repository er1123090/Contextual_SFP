"""Utilities for extracting contextual information from CUPID prior interactions.

This script reads a Parquet dataset containing CUPID evaluation rows, sends each
`prior_interactions` entry to the OpenAI GPT-4o Mini model, and appends the
structured extraction to a new column in the exported dataset.

Example usage::

    python scripts/extract_cupid_contextual_info.py \
        --input /data/minseo/Contextual_SFP_extract/data/CUPID/test.parquet \
        --output ./data/CUPID/test_with_contextual_info.parquet

The script requires the ``OPENAI_API_KEY`` environment variable to be set (or an
API key passed explicitly through ``--api-key``).
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Union
import pandas as pd
from openai import OpenAI


LOGGER = logging.getLogger(__name__)

_DEFAULT_MODEL = "gpt-4o-mini"
_SYSTEM_PROMPT = ("""
    "You are an expert analyst who extracts structured contextual information from "
    "dialogues. Each prior interaction contains exactly one contextual factor and one "
    "contextual preference. Identify the real-world situation being discussed, the "
    "contextual factor, and the contextual preference. "
    "Contextual Situation (S): The overall spatiotemporal and social context in which the utterance occurs"
    "Contextual Factor (F): The specific object or attribute that directly influences the preference"
    "Preference (P): The expressed preference or attitude"
    "Provide concise paraphrases "
    "grounded in the supplied dialogue."

""")
        

@dataclass
class ExtractionConfig:
    """Configuration used for GPT-4o Mini extraction."""

    model: str = _DEFAULT_MODEL
    max_retries: int = 3
    sleep: float = 0.0


def _normalise_prior_interactions(value: Any) -> str:
    """Convert the ``prior_interactions`` column into a serialised string.

    The Parquet file sometimes stores rich objects (lists or dicts). To ensure a
    consistent prompt, the value is converted into a JSON-formatted string when
    possible. If the conversion fails, ``str(value)`` is returned.
    """

    if value is None:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, (list, tuple)):
        try:
            return json.dumps(value, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            return str(value)

    if isinstance(value, dict):
        try:
            return json.dumps(value, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            return str(value)

    # Attempt to parse strings that may have been serialised with single quotes
    if isinstance(value, (bytes, bytearray)):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.decode("utf-8", errors="replace")

    return str(value)


def _maybe_deserialise(raw: str) -> Union[str, List[Any]]:
    """Attempt to deserialise a string into Python objects for GPT hints.

    If ``raw`` can be parsed as JSON or via :func:`ast.literal_eval`, the parsed
    value is returned. Otherwise, the original string is yielded.
    """

    stripped = raw.strip()
    if not stripped:
        return ""

    if stripped[0] in "[{":
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(stripped)
            except (ValueError, SyntaxError):
                return raw
    return raw


def _prepare_prompt_content(prior_value: Any) -> str:
    """Build the text sent to GPT based on the prior interaction."""

    normalised = _normalise_prior_interactions(prior_value)
    parsed = _maybe_deserialise(normalised)

    if isinstance(parsed, (list, tuple, dict)):
        try:
            return json.dumps(parsed, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            return normalised

    return normalised


def _extract_with_retries(
    client: OpenAI,
    prompt_text: str,
    *,
    config: ExtractionConfig,
) -> Optional[Dict[str, Any]]:
    """Call GPT-4o Mini with retries and exponential back-off."""

    if not prompt_text.strip():
        return {"sessions": []}

    user_instructions = ("""
        "You are given the `interaction session` content for a persona. "
        "Each entry corresponds to a single session containing one contextual "
        "factor and one contextual preference. Identify each session and capture "
        "the contextual situation, factor, and preference. Return the results as "
        "JSON matching the provided schema."
                         
        "For example  "

        "Contextual Situation (S): The overall spatiotemporal and social context in which the utterance occurs"
        "Contextual Factor (F): The specific object or attribute that directly influences the preference"
        "Preference (P): The expressed preference or attitude"
        "Utterance: “I’ve been drinking less coffee lately.”"
        "Expected Output:"
        "S: Recent time period (temporal context)"
        "F: Coffee (beverage entity)"
        "P: reduced consumption"

        "Provide concise paraphrases "
        "grounded in the supplied dialogue."

        "[output format]"
                        "properties": {
                            "contextual_situation": {
                                "type": "string",
                                "description": (
                                    "A short description of the real-world situation or scenario "
                                    "derived from the dialogue."
                                ),
                            },
                            "contextual_factor": {
                                "type": "string",
                                "description": (
                                    "The contextual factor explicitly mentioned in the dialogue "
                                    "(e.g., location, timing, participants)."
                                ),
                            },
                            "contextual_preference": {
                                "type": "string",
                                "description": (
                                    "The participant preference or requirement linked to the "
                                    "contextual factor."
                                ),
                            },
                            "supporting_dialogue": {
                                "type": ["string", "null"],
                                "description": (
                                    "Direct quote or concise snippet from the dialogue that "
                                    "justifies the extraction."
                                ),
                                "default": None,
                            },
                        }

    """)

    for attempt in range(1, config.max_retries + 1):
        try:
            response = client.responses.create(
                model=config.model,
                input=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_instructions},
                            {
                                "type": "text",
                                "text": "Prior interactions data:\n" + prompt_text,
                            },
                        ],
                    },
                ]
            )

            # The Responses API exposes the structured content in ``output_text``.
            raw_text = getattr(response, "output_text", None)
            if not raw_text and getattr(response, "output", None):
                # Fall back to manual reconstruction.
                content_blocks = []
                for block in response.output or []:
                    for item in block.content or []:
                        if getattr(item, "type", None) == "output_text":
                            content_blocks.append(item.text)
                        elif getattr(item, "type", None) == "text":
                            content_blocks.append(item.text)
                raw_text = "".join(content_blocks)

            if not raw_text:
                raise ValueError("Empty response received from the model")

            parsed = json.loads(raw_text)
            if config.sleep:
                time.sleep(config.sleep)
            return parsed
        except Exception as exc:  # pragma: no cover - runtime safety
            wait_time = min(2 ** (attempt - 1), 30)
            LOGGER.warning(
                "Attempt %s failed: %s", attempt, exc, exc_info=False
            )
            if attempt == config.max_retries:
                LOGGER.error("Giving up after %s attempts", attempt)
                return None
            time.sleep(wait_time)
    return None


def extract_contextual_information(
    df: "pd.DataFrame",
    client: OpenAI,
    *,
    config: ExtractionConfig,
    progress_every: int = 25,
) -> List[Optional[Dict[str, Any]]]:
    """Iterate over the dataframe and gather GPT-4o Mini outputs."""

    results: List[Optional[Dict[str, Any]]] = []
    for index, value in enumerate(df["prior_interactions"]):
        prompt = _prepare_prompt_content(value)
        extraction = _extract_with_retries(client, prompt, config=config)
        results.append(extraction)

        if (index + 1) % progress_every == 0:
            LOGGER.info("Processed %d/%d rows", index + 1, len(df))
    return results


def _configure_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_arguments(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract contextual situation, factor, and preference for each "
            "prior interaction session in the CUPID dataset using GPT-4o Mini."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input Parquet file (e.g., test.parquet).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path where the enriched Parquet file will be written.",
    )
    parser.add_argument(
        "--model",
        default=_DEFAULT_MODEL,
        help=f"OpenAI model to use (default: {_DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retries per request (default: 3).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional delay (seconds) after each successful call to respect rate limits.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenAI API key to use. Defaults to the OPENAI_API_KEY environment variable.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging output.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_arguments(argv)
    _configure_logging(args.verbose)

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        LOGGER.error("An OpenAI API key must be provided via --api-key or OPENAI_API_KEY.")
        return 1

    os.environ.setdefault("OPENAI_API_KEY", api_key)
    client = OpenAI(api_key=api_key)

    LOGGER.info("Loading Parquet data from %s", args.input)
    df = pd.read_parquet(args.input)

    config = ExtractionConfig(model=args.model, max_retries=args.max_retries, sleep=args.sleep)
    LOGGER.info(
        "Starting contextual extraction with model %s over %d rows", config.model, len(df)
    )

    extraction_results = extract_contextual_information(df, client, config=config)

    # Store both the parsed dictionary and a JSON string for convenience.
    df["gpt4o_mini_contextual_analysis"] = extraction_results
    df["gpt4o_mini_contextual_analysis_json"] = [
        json.dumps(result, ensure_ascii=False) if result is not None else None
        for result in extraction_results
    ]

    LOGGER.info("Writing enriched dataset to %s", args.output)
    df.to_parquet(args.output, index=False)
    LOGGER.info("Completed contextual extraction.")
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    sys.exit(main())
