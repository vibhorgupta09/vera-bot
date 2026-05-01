"""Standalone compose contract for the Magic Pin Vera challenge.

The HTTP service uses app.py, but the challenge brief also asks for a simple
importable module exposing compose(category, merchant, trigger, customer).
This wrapper reuses the same Composer implementation so the static JSONL, API,
and import-based checks all produce the same style of message.
"""

from pathlib import Path
from typing import Optional

from category_store import CategoryStore
from composer import Composer
from validator import Validator


ROOT_DIR = Path(__file__).resolve().parent
_category_store = CategoryStore(str(ROOT_DIR / "dataset" / "categories"))
_composer = Composer(_category_store)
_validator = Validator()


def compose(category: dict, merchant: dict, trigger: dict, customer: Optional[dict] = None) -> dict:
    action, rationale = _composer.compose(category, merchant, trigger, customer)
    valid, problems = _validator.validate(action, category, merchant, trigger, customer)
    if not valid:
        body = "Unable to compose safely from the provided context."
        rationale = "Validation blocked composition: " + "; ".join(problems)
        return {
            "body": body,
            "cta": "follow_up",
            "send_as": action.get("send_as", "vera"),
            "suppression_key": action.get("suppression_key", trigger.get("suppression_key", "")),
            "rationale": rationale,
        }

    return {
        "body": action["body"],
        "cta": action["cta"],
        "send_as": action["send_as"],
        "suppression_key": action["suppression_key"],
        "rationale": rationale,
    }
