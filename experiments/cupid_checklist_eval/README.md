# CUPID Checklist Evaluation

This experiment evaluates how well a GPT model (default: `gpt-4o-mini`) answers
the contextual checklist items contained in `data/CUPID/test.json`.

## Usage

```bash
python -m experiments.cupid_checklist_eval.evaluate_checklist --dry-run
```

The `--dry-run` flag skips API calls and assumes that every checklist item is
answered with "yes". Remove the flag to perform real evaluations (requires the
`openai` Python package and a valid `OPENAI_API_KEY`).

Use `--max-samples` to limit the number of evaluated records.
