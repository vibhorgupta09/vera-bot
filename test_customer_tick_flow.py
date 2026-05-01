#!/usr/bin/env python3
"""Smoke tests for customer-scoped /tick composition.

These run the composer directly with LLM disabled so the deterministic fallback
and customer/merchant routing stay testable without network calls.
"""

import json
import os
from pathlib import Path

os.environ["VERA_USE_LLM"] = "0"

from category_store import CategoryStore
from composer import Composer
from validator import Validator


ROOT = Path(__file__).resolve().parent


def load_seed(name):
    with open(ROOT / "dataset" / name, "r", encoding="utf-8") as handle:
        return json.load(handle)


def index_by(items, key):
    return {item[key]: item for item in items}


def assert_customer_action(label, action, rationale, category, merchant, trigger, customer):
    valid, problems = Validator().validate(action, category, merchant, trigger, customer)
    body = action.get("body", "")
    forbidden = ["vera", "want me to draft", "approval draft", "trigger", "payload"]
    for phrase in forbidden:
        if phrase in body.lower():
            problems.append(f"forbidden customer phrase: {phrase}")
    if action.get("send_as") != "merchant_on_behalf":
        problems.append(f"expected merchant_on_behalf, got {action.get('send_as')}")
    if action.get("customer_id") != customer.get("customer_id"):
        problems.append("customer_id was not preserved")
    if not body.strip().endswith("?"):
        problems.append("customer body should end with a question")
    status = "PASS" if valid and not problems else "FAIL"
    print(f"\n[{status}] {label}")
    print(f"body: {body}")
    print(f"cta:  {action.get('cta')}")
    print(f"why:  {rationale}")
    if problems:
        print("problems:", "; ".join(problems))
    return valid and not problems


def main():
    categories = CategoryStore(str(ROOT / "dataset" / "categories"))
    merchants = index_by(load_seed("merchants_seed.json")["merchants"], "merchant_id")
    customers = index_by(load_seed("customers_seed.json")["customers"], "customer_id")
    triggers = load_seed("triggers_seed.json")["triggers"]
    composer = Composer(categories)

    ok = True
    customer_triggers = [trigger for trigger in triggers if trigger.get("scope") == "customer"]
    for trigger in customer_triggers:
        merchant = merchants[trigger["merchant_id"]]
        customer = customers[trigger["customer_id"]]
        category = categories.get_category(merchant.get("category_slug"))
        action, rationale = composer.compose(category, merchant, trigger, customer)
        ok = assert_customer_action(trigger["kind"], action, rationale, category, merchant, trigger, customer) and ok

    base = customer_triggers[0]
    unknown = dict(base)
    unknown["id"] = "trg_unknown_customer_check"
    unknown["kind"] = "post_visit_instruction_due"
    unknown["payload"] = {"service": "post_cleaning_check", "date": "2026-11-13"}
    unknown["suppression_key"] = "unknown_customer:c_001"
    merchant = merchants[unknown["merchant_id"]]
    customer = customers[unknown["customer_id"]]
    category = categories.get_category(merchant.get("category_slug"))
    action, rationale = composer.compose(category, merchant, unknown, customer)
    ok = assert_customer_action("unknown customer fallback", action, rationale, category, merchant, unknown, customer) and ok

    no_consent_customer = dict(customer)
    no_consent_customer["preferences"] = dict(no_consent_customer.get("preferences", {}), reminder_opt_in=False)
    no_consent_customer["consent"] = {"opted_in_at": None, "scope": []}
    action, _ = composer.compose(category, merchant, base, no_consent_customer)
    no_send_ok = not action.get("body")
    print(f"\n[{'PASS' if no_send_ok else 'FAIL'}] no consent skip")
    if action.get("body"):
        print(f"body: {action.get('body')}")
    ok = no_send_ok and ok

    print(f"\nSummary: {'PASS' if ok else 'FAIL'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
