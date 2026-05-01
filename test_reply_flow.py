#!/usr/bin/env python3
"""Local smoke test for /v1/reply decision logic.

This script imports the reply handler directly and exercises deterministic
merchant/customer reply paths without starting Flask or calling an LLM.
"""

import os

os.environ["VERA_USE_LLM"] = "0"

from conversation_handlers import respond
from validator import Validator


MERCHANT = {
    "merchant_id": "m_test",
    "category_slug": "restaurants",
    "identity": {
        "name": "Test Cafe",
        "owner_first_name": "Asha",
        "locality": "Saket",
        "city": "Delhi",
    },
    "performance": {
        "views": 1200,
        "calls": 44,
        "ctr": 0.04,
        "delta_7d": {"calls_pct": 0.12},
    },
    "offers": [{"title": "Lunch Combo @ Rs 149", "status": "active"}],
}

CATEGORY = {
    "slug": "restaurants",
    "voice": {
        "tone": "warm_busy_practical",
        "code_mix": "hindi_english_natural",
        "vocab_allowed": [],
        "vocab_taboo": [],
    },
}

TRIGGER = {
    "id": "trg_test",
    "scope": "merchant",
    "kind": "active_planning_intent",
    "urgency": 4,
    "payload": {"intent_topic": "corporate_lunch"},
}

CUSTOMER = {
    "customer_id": "c_test",
    "identity": {"name": "Ravi"},
    "state": "active",
}


def base_state(customer=False):
    trigger = dict(TRIGGER)
    state = {
        "conversation_id": "conv_test",
        "merchant_id": MERCHANT["merchant_id"],
        "merchant": MERCHANT,
        "category": CATEGORY,
        "trigger": trigger,
        "trigger_kind": trigger["kind"],
        "sent_bodies": ["Asha, corporate lunch is picking up. Want me to draft the menu?"],
        "turns": [],
    }
    if customer:
        trigger["scope"] = "customer"
        state["customer_id"] = CUSTOMER["customer_id"]
        state["customer"] = CUSTOMER
    return state


def run_case(label, message, state, expected_action=None):
    state = dict(state)
    state["turns"] = list(state.get("turns", [])) + [{"from": "customer" if state.get("customer") else "merchant", "message": message}]
    decision = respond(state, message)
    valid, problems = Validator().validate_reply(decision, state)
    if expected_action and decision.get("action") != expected_action:
        valid = False
        problems = list(problems) + [f"expected action {expected_action}, got {decision.get('action')}"]
    status = "PASS" if valid else "FAIL"
    print(f"\n[{status}] {label}")
    print(f"in:  {message}")
    print(f"out: {decision}")
    if problems:
        print("problems:", "; ".join(problems))
    return valid


def main():
    cases = [
        ("hostile", "Stop messaging me, this is spam", base_state(), "end"),
        ("auto reply", "Thank you for contacting us. Our team will respond shortly.", base_state(), "wait"),
        ("commitment", "Ok lets do it. What's next?", base_state(), "send"),
        ("pause", "Not now, maybe next week", base_state(), "wait"),
        ("sample", "Can you show me a sample first?", base_state(), "send"),
        ("edit", "Make it shorter and remove discount", base_state(), "send"),
        ("price objection", "This is too expensive for us", base_state(), "send"),
        ("proof question", "Will this work? What proof do you have?", base_state(), "send"),
        ("already done", "I already posted this yesterday", base_state(), "send"),
        ("mixed yes pause", "Yes, but not now. Maybe next week.", base_state(), "wait"),
        ("mixed yes price", "Ok, but this is too expensive for us", base_state(), "send"),
        ("sample dont send", "Show me a sample first, don't send anything", base_state(), "send"),
        ("angry question", "Why are you bothering me with this?", base_state(), "send"),
        ("wrong number", "Wrong number, stop", base_state(), "end"),
        ("off topic", "Can you help with GST filing?", base_state(), "send"),
        ("detail", "What exactly will you write?", base_state(), "send"),
        ("customer confirm", "Yes, works for me", base_state(customer=True), "send"),
        ("customer change", "I need another slot", base_state(customer=True), "send"),
        ("customer detail", "What is the price?", base_state(customer=True), "send"),
        ("customer optout", "Wrong number, stop messaging", base_state(customer=True), "end"),
    ]
    ok = True
    for label, message, state, expected_action in cases:
        ok = run_case(label, message, state, expected_action) and ok
    print(f"\nSummary: {'PASS' if ok else 'FAIL'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
