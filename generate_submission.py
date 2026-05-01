#!/usr/bin/env python3
"""Generate submission.jsonl for the Magic Pin Vera challenge.

The seed dataset only contains 25 representative triggers. The challenge
generator expands it to 100 triggers and 30 canonical pairs, so this script uses
the same deterministic generator in memory and writes exactly 30 validated rows
when enough valid contexts are available.
"""

from __future__ import annotations

import importlib.util
import json
import os
import random
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
DATASET_DIR = ROOT_DIR / "dataset"
OUTPUT_FILE = ROOT_DIR / "submission.jsonl"


def load_dotenv():
    env_file = ROOT_DIR / ".env"
    if not env_file.exists():
        return
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


# Keep environment loading before imports that may read API/model settings.
load_dotenv()

from category_store import CategoryStore
from composer import Composer
from validator import Validator


def import_dataset_generator():
    generator_path = DATASET_DIR / "generate_dataset.py"
    spec = importlib.util.spec_from_file_location("vera_dataset_generator", generator_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_expanded_dataset():
    generator = import_dataset_generator()
    rnd = random.Random(generator.SEED)
    categories, merchant_seeds, customer_seeds, trigger_seeds = generator.load_seeds(DATASET_DIR)
    merchants = generator.expand_merchants(merchant_seeds, rnd)
    customers = generator.expand_customers(customer_seeds, merchants, rnd)
    triggers = generator.expand_triggers(trigger_seeds, merchants, customers, rnd)
    return categories, merchants, customers, triggers


def build_candidate_pairs(triggers):
    """Mirror dataset/generate_dataset.py, then append fallbacks."""
    by_kind = {}
    for trigger in triggers:
        by_kind.setdefault(trigger.get("kind", "unknown"), []).append(trigger)

    pairs = []
    seen = set()
    for kind in sorted(by_kind):
        for trigger in by_kind[kind][:2]:
            key = (trigger.get("merchant_id"), trigger.get("id"))
            if key in seen:
                continue
            pairs.append({
                "merchant_id": trigger.get("merchant_id"),
                "trigger_id": trigger.get("id"),
                "customer_id": trigger.get("customer_id"),
            })
            seen.add(key)
            if len(pairs) >= 30:
                break
        if len(pairs) >= 30:
            break

    for trigger in triggers:
        key = (trigger.get("merchant_id"), trigger.get("id"))
        if key in seen:
            continue
        pairs.append({
            "merchant_id": trigger.get("merchant_id"),
            "trigger_id": trigger.get("id"),
            "customer_id": trigger.get("customer_id"),
        })
        seen.add(key)
    return pairs


def generate_submission():
    categories, merchants, customers, triggers = build_expanded_dataset()
    merchants_by_id = {m["merchant_id"]: m for m in merchants}
    customers_by_id = {c["customer_id"]: c for c in customers}
    triggers_by_id = {t["id"]: t for t in triggers}

    category_store = CategoryStore(str(DATASET_DIR / "categories"))
    composer = Composer(category_store)
    validator = Validator()

    results = []
    skipped = []
    for pair in build_candidate_pairs(triggers):
        if len(results) >= 30:
            break

        merchant = merchants_by_id.get(pair.get("merchant_id"))
        trigger = triggers_by_id.get(pair.get("trigger_id"))
        if not merchant or not trigger:
            skipped.append((pair.get("trigger_id"), "missing merchant or trigger"))
            continue

        customer = customers_by_id.get(pair.get("customer_id")) if pair.get("customer_id") else None
        if trigger.get("scope") == "customer" and trigger.get("customer_id") and not customer:
            skipped.append((trigger.get("id"), "missing customer"))
            continue

        category = categories.get(merchant.get("category_slug")) or category_store.get_category(merchant.get("category_slug"))
        if not category:
            skipped.append((trigger.get("id"), "missing category"))
            continue

        action, rationale = composer.compose(category, merchant, trigger, customer)
        valid, problems = validator.validate(action, category, merchant, trigger, customer)
        if not valid:
            skipped.append((trigger.get("id"), "; ".join(problems)))
            continue

        results.append({
            "test_id": f"T{len(results) + 1:02d}",
            "body": action["body"],
            "cta": action["cta"],
            "send_as": action["send_as"],
            "suppression_key": action["suppression_key"],
            "rationale": rationale,
        })

    with OUTPUT_FILE.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Generated {len(results)}/30 valid outputs.")
    print(f"Saved to {OUTPUT_FILE.name}")
    if skipped:
        print(f"Skipped {len(skipped)} candidate pairs.")
        for trigger_id, reason in skipped[:8]:
            print(f"  {trigger_id}: {reason}")
    if len(results) != 30:
        print("Warning: fewer than 30 valid outputs were available.")

    for result in results[:3]:
        print(f"\n{result['test_id']}: {result['body'][:160]}...")


if __name__ == "__main__":
    generate_submission()
