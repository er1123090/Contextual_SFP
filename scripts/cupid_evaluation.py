"""Core evaluation routines for CUPID checklist compliance."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from openai import OpenAI

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = (
    "You are a careful analyst that determines whether a response will comply with "
    "a persona's contextual checklist. Use the prior interaction history to inform "
    "your decision and respond exactly with the requested JSON format."
)
USER_GUIDANCE = (
    "You will receive prior interaction history along with the current request, "
    "context factor, and checklist. For each checklist item, answer with 'Yes' if "
    "the response to the current request should comply with the preference implied by "
    "the context factor, otherwise answer 'No'. Return ONLY valid JSON using the "
    "schema:\n{\n  \"answers\": [\"Yes\" or \"No\" for each checklist item in order]\n}\n"
)


@dataclass
class EvaluationConfig:
    """Runtime configuration for model calls."""

    model: str = DEFAULT_MODEL
    max_retries: int = 3
    retry_sleep: float = 1.0


def load_json_dataset(path: str) -> List[Dict[str, Any]]:
    """Load the CUPID dataset from ``path``."""

    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def format_dialogue(dialogue: Iterable[Dict[str, Any]]) -> str:
    """Render a dialogue turn-by-turn for inclusion in the prompt."""

    lines: List[str] = []
    for message in dialogue:
        role = message.get("role", "user")
        content = message.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        lines.append(f"{role.capitalize()}: {content}")
    return "\n".join(lines)


def prepare_history(prior_interactions: Any) -> str:
    """Convert prior interactions into a prompt-ready string."""

    if not prior_interactions:
        return "No prior interactions provided."

    sections: List[str] = []
    for interaction in prior_interactions:
        dialogue = interaction.get("dialogue") if isinstance(interaction, dict) else None
        if dialogue:
            sections.append(format_dialogue(dialogue))
    return "\n\n".join(sections) if sections else "No usable dialogue found."


def build_user_prompt(sample: Dict[str, Any]) -> str:
    """Create the user prompt content for a dataset sample."""

    checklist = sample.get("current_checklist") or []
    checklist_lines = "\n".join(
        f"{idx + 1}. {item}" for idx, item in enumerate(checklist)
    )

    parts = [
        "Current request:\n" + str(sample.get("current_request", "")),
        "\nCurrent context factor:\n" + str(sample.get("current_context_factor", "")),
        "\nChecklist items:\n" + (checklist_lines or "(none provided)"),
    ]
    return "".join(parts)


def _call_model(
    client: OpenAI,
    *,
    history: str,
    prompt: str,
    config: EvaluationConfig,
) -> Optional[Dict[str, Any]]:
    last_error: Optional[Exception] = None
    for attempt in range(1, config.max_retries + 1):
        try:
            response = client.responses.create(
                model=config.model,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": USER_GUIDANCE},
                            {"type": "text", "text": "Prior interaction history:\n" + history},
                            {"type": "text", "text": prompt},
                        ],
                    },
                ],
            )

            raw_text = getattr(response, "output_text", None)
            if not raw_text and getattr(response, "output", None):
                chunks: List[str] = []
                for block in response.output or []:
                    for item in block.content or []:
                        if getattr(item, "type", None) in {"output_text", "text"}:
                            chunks.append(item.text)
                raw_text = "".join(chunks)

            if not raw_text:
                raise ValueError("Empty response from model")

            return json.loads(raw_text)
        except Exception as exc:  # pragma: no cover - runtime network handling
            last_error = exc
            LOGGER.warning("Attempt %s failed: %s", attempt, exc)
            if attempt < config.max_retries:
                time.sleep(config.retry_sleep)
    LOGGER.error("Model call failed after %s attempts: %s", config.max_retries, last_error)
    return None


def evaluate_dataset(
    dataset: List[Dict[str, Any]],
    *,
    client: OpenAI,
    config: EvaluationConfig,
) -> Dict[str, Any]:
    """Run evaluation over the dataset and compute accuracy."""

    total_items = 0
    correct_items = 0
    per_sample: List[Dict[str, Any]] = []

    for sample in dataset:
        history = prepare_history(sample.get("prior_interactions"))
        prompt = build_user_prompt(sample)
        prediction = _call_model(client, history=history, prompt=prompt, config=config)

        checklist = sample.get("current_checklist") or []
        answers = []
        if prediction and isinstance(prediction, dict):
            answers = prediction.get("answers") or []

        normalised = [str(ans).strip().lower() for ans in answers]
        for idx, _ in enumerate(checklist):
            pred_label = normalised[idx] if idx < len(normalised) else ""
            total_items += 1
            if pred_label.startswith("y"):
                correct_items += 1

        per_sample.append(
            {
                "persona_id": sample.get("persona_id"),
                "current_context_factor": sample.get("current_context_factor"),
                "checklist": checklist,
                "answers": answers,
                "raw_prediction": prediction,
            }
        )

    accuracy = correct_items / total_items if total_items else 0.0
    return {
        "accuracy": accuracy,
        "total_items": total_items,
        "correct_items": correct_items,
        "predictions": per_sample,
    }


__all__ = [
    "DEFAULT_MODEL",
    "SYSTEM_PROMPT",
    "USER_GUIDANCE",
    "EvaluationConfig",
    "build_user_prompt",
    "evaluate_dataset",
    "format_dialogue",
    "load_json_dataset",
    "prepare_history",
]
