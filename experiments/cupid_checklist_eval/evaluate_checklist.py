"""Evaluate GPT models on CUPID checklist adherence.

This script loads the CUPID test set and, for each sample, builds a
conversation history using the prior interactions. The current context factor
and checklist are provided as the final query and the model (default
``gpt-4o-mini``) is asked to answer each checklist item with a ``yes`` or ``no``.

The gold label for each checklist item is ``yes``. Accuracy is computed as the
fraction of checklist items for which the model responds with ``yes``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:  # pragma: no cover - the OpenAI SDK may not be installed in all envs
    from openai import OpenAI
except Exception as exc:  # pragma: no cover - handled gracefully at runtime
    OpenAI = None  # type: ignore
    _IMPORT_ERROR = exc
else:  # pragma: no cover - executed only when the SDK is present
    _IMPORT_ERROR = None

LOGGER = logging.getLogger(__name__)


@dataclass
class CupidSample:
    """Container for a single CUPID sample."""

    persona_id: str
    current_request: str
    current_context_factor: str
    current_contextual_preference: str
    current_checklist: Sequence[str]
    prior_dialogues: Sequence[Sequence[Dict[str, str]]]

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "CupidSample":
        prior_dialogues: List[List[Dict[str, str]]] = []
        for interaction in payload.get("prior_interactions", []):
            dialogue = interaction.get("dialogue", [])
            if dialogue:
                prior_dialogues.append(dialogue)
        return cls(
            persona_id=payload.get("persona_id", ""),
            current_request=payload.get("current_request", ""),
            current_context_factor=payload.get("current_context_factor", ""),
            current_contextual_preference=payload.get(
                "current_contextual_preference", ""
            ),
            current_checklist=payload.get("current_checklist", []),
            prior_dialogues=prior_dialogues,
        )


def iter_samples(dataset_path: Path) -> Iterable[CupidSample]:
    """Yield :class:`CupidSample` objects from the JSON dataset."""

    with dataset_path.open("r", encoding="utf-8") as fh:
        raw_data = json.load(fh)

    for entry in raw_data:
        yield CupidSample.from_dict(entry)


def build_prompt(sample: CupidSample) -> List[Dict[str, str]]:
    """Construct a chat prompt incorporating the prior dialogue and query."""

    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are an assistant that evaluates whether a response follows "
                "specific contextual preferences. For the final user query, "
                "reply strictly in JSON with the schema: {\"results\": ["
                "{\"checklist_item\": str, \"answer\": \"yes\"|\"no\"} ... ]}."
            ),
        }
    ]

    for dialogue in sample.prior_dialogues:
        for turn in dialogue:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if not content:
                continue
            messages.append({"role": role, "content": content})

    checklist_lines = "\n".join(
        f"- {item}" for item in sample.current_checklist
    ) or "-"

    final_prompt = (
        "Current context factor: {factor}\n"
        "Contextual preference: {preference}\n"
        "Checklist to evaluate:\n{checklist}\n\n"
        "Respond with a JSON object following the requested schema. For each "
        "checklist item, indicate \"yes\" if the preference should be satisfied "
        "based on the context factor, otherwise \"no\"."
    ).format(
        factor=sample.current_context_factor,
        preference=sample.current_contextual_preference,
        checklist=checklist_lines,
    )

    messages.append({"role": "user", "content": final_prompt})
    return messages


def extract_results(model_output: str, expected_items: Sequence[str]) -> List[str]:
    """Parse the model output and align answers with the checklist."""

    try:
        parsed = _extract_first_json_value(model_output)
    except ValueError as exc:
        raise ValueError(f"Unable to parse model output as JSON: {model_output!r}") from exc

    if isinstance(parsed, dict):
        results = parsed.get("results")
    elif isinstance(parsed, list):
        results = parsed
    else:
        raise ValueError(f"Unexpected JSON structure: {type(parsed)!r}")

    if not isinstance(results, list):
        raise ValueError("Parsed JSON does not contain a list of results.")

    answers: List[str] = []
    for idx, checklist_item in enumerate(expected_items):
        if idx < len(results):
            result = results[idx]
        else:
            result = {}
        if isinstance(result, dict):
            answer = str(result.get("answer", "")).strip().lower()
        else:
            answer = str(result).strip().lower()
        answers.append(answer)
    return answers


def _extract_first_json_value(text: str) -> Any:
    """Extract the first JSON value embedded within ``text``."""

    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(text):
        char = text[idx]
        if char in "[{":
            try:
                parsed, end = decoder.raw_decode(text[idx:])
            except json.JSONDecodeError:
                idx += 1
                continue
            return parsed
        idx += 1
    raise ValueError("No JSON object or array found in the text.")


def run_evaluation(
    dataset_path: Path,
    model: str,
    max_samples: Optional[int] = None,
    temperature: float = 0.0,
    dry_run: bool = False,
) -> None:
    """Evaluate the model and print accuracy statistics."""

    samples = iter_samples(dataset_path)
    if max_samples is not None:
        samples = _take(samples, max_samples)

    total_items = 0
    correct_items = 0

    client: Optional[OpenAI]
    if dry_run:
        client = None
    else:
        if OpenAI is None:
            raise RuntimeError(
                "The openai package is not available. Install it to run evaluations"
            ) from _IMPORT_ERROR
        client = OpenAI()

    for index, sample in enumerate(samples, start=1):
        LOGGER.info("Evaluating sample %s (%s)", index, sample.persona_id)
        messages = build_prompt(sample)

        if dry_run:
            LOGGER.debug("Dry run enabled; skipping API call.")
            model_output = json.dumps(
                {
                    "results": [
                        {"checklist_item": item, "answer": "yes"}
                        for item in sample.current_checklist
                    ]
                }
            )
        else:
            response = client.responses.create(  # type: ignore[union-attr]
                model=model,
                input=messages,
                temperature=temperature,
            )
            model_output = _collect_text_from_response(response)

        answers = extract_results(model_output, sample.current_checklist)
        for answer in answers:
            total_items += 1
            if answer.startswith("y"):
                correct_items += 1

    accuracy = correct_items / total_items if total_items else 0.0
    print(json.dumps({"total_items": total_items, "correct_items": correct_items, "accuracy": accuracy}, indent=2))


def _collect_text_from_response(response: Any) -> str:
    """Collect concatenated text from an OpenAI responses API payload."""

    content_chunks: List[str] = []
    for item in getattr(response, "output", []):
        if isinstance(item, dict):
            item_type = item.get("type")
            if item_type == "message":
                for part in item.get("content", []):
                    if part.get("type") == "output_text":
                        content_chunks.append(part.get("text", ""))
            elif item_type == "output_text":
                content_chunks.append(item.get("text", ""))
        elif hasattr(item, "content"):  # pragma: no cover - depends on SDK version
            content_chunks.append(str(item.content))
    if not content_chunks and hasattr(response, "output_text"):
        content_chunks.append(str(response.output_text))
    if not content_chunks and hasattr(response, "choices"):
        # Compatibility with chat.completions responses
        choices = getattr(response, "choices")
        for choice in choices:
            message = getattr(choice, "message", None)
            if message and hasattr(message, "content"):
                content_chunks.append(str(message.content))
    return "\n".join(content_chunks).strip()


def _take(iterable: Iterable[CupidSample], limit: int) -> Iterable[CupidSample]:
    """Yield at most ``limit`` items from ``iterable``."""

    count = 0
    for item in iterable:
        if count >= limit:
            break
        yield item
        count += 1


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/CUPID/test.json"),
        help="Path to the CUPID dataset JSON file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model name to evaluate.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional maximum number of samples to evaluate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the model.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip API calls and assume every answer is 'yes'. Useful for testing.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        help="Logging level (e.g., INFO, DEBUG).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for command line execution."""

    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.WARNING))

    try:
        run_evaluation(
            dataset_path=args.dataset,
            model=args.model,
            max_samples=args.max_samples,
            temperature=args.temperature,
            dry_run=args.dry_run,
        )
    except Exception as exc:  # pragma: no cover - CLI error handling
        LOGGER.error("Evaluation failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI invocation guard
    sys.exit(main())
