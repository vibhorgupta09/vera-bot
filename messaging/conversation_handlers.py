"""Small deterministic reply handler for replay-style judge conversations.

The challenge scores multi-turn handling as a tiebreaker. This module keeps that
logic separate from Flask so the same behaviour can be imported directly if the
submission reviewer asks for the optional conversation handler artifact.
"""

import json
import os
import re

from llm.llm_client import LLMClient


AUTO_REPLY_PATTERNS = [
    "thank you for contacting",
    "thanks for contacting",
    "our team will respond",
    "will respond shortly",
    "automated assistant",
    "auto-reply",
    "business hours",
    "we will get back",
]

HOSTILE_OR_OPTOUT_PATTERNS = [
    "stop messaging",
    "stop sending",
    "not interested",
    "unsubscribe",
    "do not message",
    "do not contact",
    "dont message",
    "don't message",
    "don't contact",
    "wrong number",
    "remove my number",
    "useless spam",
    "spam",
]

COMMITMENT_PATTERNS = [
    "lets do it",
    "let's do it",
    "go ahead",
    "proceed",
    "confirm",
    "do it",
    "what's next",
    "whats next",
    "start it",
    "i want to join",
    "mujhe magicpin",
    "judrna",
    "jodna",
    "join magicpin",
]

OFF_TOPIC_TERMS = ["gst", "tax filing", "income tax", "itr", "loan", "accounting"]
PAUSE_PATTERNS = [
    "not now", "later", "next week", "tomorrow", "kal", "baad mein", "bad mein",
    "busy", "no time", "abhi nahi", "abhi nahin", "not today",
]
SAMPLE_PATTERNS = ["sample", "example", "draft dikhao", "show draft", "send draft", "preview"]
EDIT_PATTERNS = ["change", "edit", "modify", "revise", "shorter", "longer", "different", "tone", "remove", "add "]
PRICE_OBJECTION_PATTERNS = [
    "too expensive", "costly", "price high", "zyada", "mehenga", "expensive",
    "budget", "discount nahi", "kitna kharcha", "how much cost",
]
PROOF_QUESTION_PATTERNS = [
    "will this work", "why should", "why do", "kya fayda", "fayda", "proof",
    "guarantee", "result", "worth it", "is this needed",
]
ALREADY_DONE_PATTERNS = [
    "already did", "already done", "already posted", "posted yesterday",
    "profile verified", "verified already", "done already", "i did this",
]
CUSTOMER_CHANGE_PATTERNS = ["change", "another slot", "reschedule", "different time", "not available", "can't come", "cant come"]
VALID_REPLY_CTAS = {
    "open_ended",
    "binary_yes_no",
    "binary_confirm_cancel",
    "multi_choice_slot",
    "none",
}
INTENT_KEYS = {
    "commitment",
    "pause",
    "sample_request",
    "edit_request",
    "price_objection",
    "proof_question",
    "already_done",
    "off_topic",
    "detail_question",
    "customer_confirm",
    "customer_change",
    "customer_detail",
    "ambiguous",
}
LOW_SIGNAL_JARGON = [
    "trigger",
    "payload",
    "suppression_key",
    "send_as",
    "template",
    "template_params",
    "metric_or_topic",
    "ctr_below_peer_median",
    "high_risk_adult_cohort",
    "delta_7d",
    "digest item",
    "signals show",
    "trg_",
    "d_2026",
    "top_item_id",
]
_LLM_CLIENT = LLMClient()


def respond(state, merchant_message):
    """Return the next reply decision for a replay turn.

    `state` is intentionally a plain dict so Flask, tests, or an offline judge can
    pass whatever conversation memory they have. Expected useful keys:
    merchant, trigger, customer, turns, sent_bodies.
    """
    state = state or {}
    text = normalize_reply(merchant_message)

    if is_hostile_or_optout(text):
        return {
            "action": "end",
            "rationale": "Merchant opted out or reacted negatively; closing without another nudge.",
        }

    if is_auto_reply(text):
        seen_count = _seen_merchant_reply_count(state, text)
        if seen_count >= 3:
            return {
                "action": "end",
                "rationale": "Same WhatsApp Business auto-reply repeated three times; no human signal.",
            }
        return {
            "action": "wait",
            "wait_seconds": 86400 if seen_count >= 2 else 14400,
            "rationale": "Detected a canned auto-reply; backing off instead of burning conversation turns.",
        }

    intent = _classify_reply(state, merchant_message, text)
    return _route_intent(state, text, merchant_message, intent)


def normalize_reply(message):
    return re.sub(r"\s+", " ", str(message or "").strip().lower())


def is_auto_reply(text):
    return any(pattern in text for pattern in AUTO_REPLY_PATTERNS)


def is_hostile_or_optout(text):
    return any(pattern in text for pattern in HOSTILE_OR_OPTOUT_PATTERNS)


def is_commitment(text):
    return any(pattern in text for pattern in COMMITMENT_PATTERNS)


def is_soft_yes(text):
    if text in {"yes", "yes please", "ok", "okay", "sure", "haan", "ha", "chalega", "works for me"}:
        return True
    return bool(re.match(r"^(yes|yeah|yep|sure|ok|okay|haan|ha)\b[\s,!.:-]*(please)?", text))


def is_off_topic(text):
    return any(term in text for term in OFF_TOPIC_TERMS)


def is_pause_request(text):
    return any(pattern in text for pattern in PAUSE_PATTERNS)


def _pause_seconds(text):
    if any(token in text for token in ["next week", "week"]):
        return 604800
    if any(token in text for token in ["tomorrow", "kal"]):
        return 86400
    return 259200


def is_sample_request(text):
    return any(pattern in text for pattern in SAMPLE_PATTERNS)


def is_edit_request(text):
    return any(pattern in text for pattern in EDIT_PATTERNS)


def is_price_objection(text):
    return any(pattern in text for pattern in PRICE_OBJECTION_PATTERNS)


def is_proof_question(text):
    return any(pattern in text for pattern in PROOF_QUESTION_PATTERNS)


def is_already_done(text):
    return any(pattern in text for pattern in ALREADY_DONE_PATTERNS)


def is_detail_question(text):
    question_terms = ["how", "what", "price", "cost", "kitna", "details", "kaise", "explain", "?"]
    return any(term in text for term in question_terms)


def _classify_reply(state, merchant_message, text):
    rule_intent = _rule_classify_reply(state, text)
    llm_intent = _llm_classify_reply(state, merchant_message)
    if not llm_intent:
        return rule_intent

    # Keep crisp local overrides for mixed-intent replies like "yes, but not now".
    if rule_intent["intent"] != "ambiguous" and rule_intent["confidence"] >= 0.9:
        if llm_intent["intent"] == "commitment" and rule_intent["intent"] != "commitment":
            return rule_intent
        if rule_intent["intent"] in {"pause", "off_topic", "customer_change", "customer_detail"}:
            return rule_intent
    return llm_intent


def _rule_classify_reply(state, text):
    customer_scope = _is_customer_scope(state)
    if customer_scope:
        if is_soft_yes(text) or is_commitment(text):
            return _intent("customer_confirm", 0.95, "Customer accepted the proposed reminder, slot, or next step.")
        if any(pattern in text for pattern in CUSTOMER_CHANGE_PATTERNS) or is_pause_request(text):
            return _intent("customer_change", 0.95, "Customer needs a slot, delivery, or reminder change.")
        if is_detail_question(text) or is_price_objection(text):
            return _intent("customer_detail", 0.9, "Customer asked for practical detail or price.")
        return _intent("ambiguous", 0.35, "Customer reply did not match a deterministic route.")

    # Mixed intent priority: objection/pause/sample beats a generic yes.
    if is_pause_request(text):
        return _intent("pause", 0.95, "Merchant asked to pause or come back later.")
    if is_off_topic(text):
        return _intent("off_topic", 0.95, "Merchant asked for non-Vera work.")
    if is_already_done(text):
        return _intent("already_done", 0.95, "Merchant said the suggested action may already be complete.")
    if is_sample_request(text):
        return _intent("sample_request", 0.9, "Merchant asked to see a draft or example.")
    if is_edit_request(text):
        return _intent("edit_request", 0.9, "Merchant requested a change to the draft.")
    if is_price_objection(text):
        return _intent("price_objection", 0.9, "Merchant raised price, cost, or budget concern.")
    if is_proof_question(text):
        return _intent("proof_question", 0.9, "Merchant asked why this is useful or whether it will work.")
    if is_commitment(text) or is_soft_yes(text):
        return _intent("commitment", 0.9, "Merchant showed intent to proceed.")
    if is_detail_question(text):
        return _intent("detail_question", 0.75, "Merchant asked for detail.")
    return _intent("ambiguous", 0.35, "Merchant reply did not match a deterministic route.")


def _intent(intent, confidence, rationale="", secondary_intent=None, scope=None):
    return {
        "intent": intent if intent in INTENT_KEYS else "ambiguous",
        "secondary_intent": secondary_intent,
        "scope": scope,
        "confidence": confidence,
        "rationale": rationale,
    }


def _llm_classify_reply(state, merchant_message):
    if not _llm_reply_ready():
        return None
    prompt = _llm_intent_prompt(state, merchant_message)
    response = _LLM_CLIENT.generate(
        prompt,
        system=(
            "You classify Magicpin Vera WhatsApp replies into fixed intent keys. "
            "Return compact JSON only. Do not draft a reply."
        ),
        max_tokens=160,
        temperature=0.0,
    )
    parsed = _parse_json(response)
    if not parsed:
        return None
    intent = str(parsed.get("intent", "ambiguous")).strip()
    secondary = parsed.get("secondary_intent")
    try:
        confidence = float(parsed.get("confidence", 0))
    except (TypeError, ValueError):
        confidence = 0
    if intent not in INTENT_KEYS or confidence < 0.65:
        return None
    rationale = re.sub(r"\s+", " ", str(parsed.get("rationale", "")).strip())
    return _intent(intent, confidence, rationale, secondary_intent=secondary, scope=parsed.get("scope"))


def _llm_intent_prompt(state, merchant_message):
    facts = _reply_fact_pack(state, merchant_message, mode="classify")
    return (
        "Classify the incoming WhatsApp reply into exactly one intent key. Do not write the reply.\n"
        "Return JSON only: {\"intent\":\"...\",\"secondary_intent\":\"...\",\"scope\":\"merchant|customer\",\"confidence\":0.0,\"rationale\":\"...\"}\n\n"
        "Intent keys:\n"
        "- commitment: user wants to proceed or asks what next.\n"
        "- pause: user says later, not now, busy, tomorrow, next week. Use pause even if they also say yes.\n"
        "- sample_request: user asks to see a sample, draft, preview, or example before sending.\n"
        "- edit_request: user asks to change wording, length, tone, offer, or remove/add something.\n"
        "- price_objection: user says expensive, budget issue, cost concern, or asks if spend is needed.\n"
        "- proof_question: user asks why this matters, whether it works, or asks for proof/results.\n"
        "- already_done: user says they already posted, verified, completed, or did the action.\n"
        "- off_topic: user asks for GST, tax, loan, accounting, or unrelated operations work.\n"
        "- detail_question: user asks what/how/details/price without a clear objection.\n"
        "- customer_confirm: customer accepts a slot, reminder, refill, booking, delivery, or next step.\n"
        "- customer_change: customer wants another slot, reschedule, delivery/prescription change, or says unavailable.\n"
        "- customer_detail: customer asks price, who this is, details, or practical info.\n"
        "- ambiguous: none of the above is clear.\n\n"
        "Rules:\n"
        "- Hostile/opt-out and auto-reply are handled before this classifier; do not use those keys.\n"
        "- For mixed intent, choose the more cautious operational intent: pause > off_topic > already_done > sample_request > edit_request > price_objection > proof_question > commitment.\n"
        "- If FACTS show customer scope, prefer customer_* keys.\n"
        "- Do not infer completed actions not stated by the user.\n\n"
        f"FACTS:\n{json.dumps(facts, ensure_ascii=False, indent=2)}"
    )


def _route_intent(state, text, merchant_message, intent_info):
    intent = intent_info.get("intent", "ambiguous")
    rationale = intent_info.get("rationale") or "Reply classified into deterministic handler."

    if intent == "customer_confirm":
        return _send_once(state, _customer_confirm_body(state), "binary_yes_no", rationale)
    if intent == "customer_change":
        return _send_once(state, _customer_change_body(state), "open_ended", rationale)
    if intent == "customer_detail":
        return _send_once(state, _customer_detail_body(state), "multi_choice_slot", rationale)
    if _is_customer_scope(state):
        llm_decision = _llm_reply(state, merchant_message, mode="customer")
        if llm_decision:
            return _send_once(state, llm_decision["body"], llm_decision["cta"], llm_decision["rationale"])
        return _send_once(state, _customer_ack_body(state), "multi_choice_slot", rationale)

    if intent == "commitment":
        return _send_once(state, _commitment_body(state), "binary_confirm_cancel", rationale)
    if intent == "pause":
        return {
            "action": "wait",
            "wait_seconds": _pause_seconds(text),
            "rationale": rationale,
        }
    if intent == "already_done":
        return _send_once(state, _already_done_body(state), "binary_yes_no", rationale)
    if intent == "sample_request":
        return _send_once(state, _sample_body(state), "binary_yes_no", rationale)
    if intent == "edit_request":
        return _send_once(state, _edit_body(state), "open_ended", rationale)
    if intent == "price_objection":
        return _send_once(state, _price_objection_body(state), "binary_yes_no", rationale)
    if intent == "proof_question":
        return _send_once(state, _proof_body(state), "binary_yes_no", rationale)
    if intent == "off_topic":
        return _send_once(state, _off_topic_body(state), "none", rationale)
    if intent == "detail_question":
        llm_decision = _llm_reply(state, merchant_message, mode="detail")
        if llm_decision:
            return _send_once(state, llm_decision["body"], llm_decision["cta"], llm_decision["rationale"])
        return _send_once(state, _detail_body(state), "binary_yes_no", rationale)

    llm_decision = _llm_reply(state, merchant_message, mode="ambiguous")
    if llm_decision:
        return _send_once(state, llm_decision["body"], llm_decision["cta"], llm_decision["rationale"])
    return _send_once(state, _ack_body(state), "binary_yes_no", rationale)


def _llm_reply(state, merchant_message, mode):
    if not _llm_reply_ready():
        return None
    prompt = _llm_reply_prompt(state, merchant_message, mode)
    response = _LLM_CLIENT.generate(
        prompt,
        system=(
            "You are Magicpin Vera handling a merchant WhatsApp reply. "
            "Return compact JSON only. Never invent facts."
        ),
        max_tokens=260,
        temperature=0.1,
    )
    parsed = _parse_json(response)
    if not parsed:
        return None

    action = parsed.get("action", "send")
    body = re.sub(r"\s+", " ", str(parsed.get("body", "")).strip())
    cta = parsed.get("cta", "binary_yes_no")
    rationale = re.sub(r"\s+", " ", str(parsed.get("rationale", "")).strip())
    if action != "send" or not body or cta not in VALID_REPLY_CTAS:
        return None
    if len(body) < 25 or len(body) > 700:
        return None
    if any(token in body.lower() for token in ["```", "json", *LOW_SIGNAL_JARGON]):
        return None
    return {
        "body": body,
        "cta": cta,
        "rationale": rationale or f"LLM handled {mode} merchant reply from grounded conversation state.",
    }


def _llm_reply_ready():
    if os.environ.get("VERA_USE_LLM", "1").lower() in {"0", "false", "no"}:
        return False
    return _LLM_CLIENT.available()


def _llm_reply_prompt(state, merchant_message, mode):
    facts = _reply_fact_pack(state, merchant_message, mode)
    return (
        "Draft the next Vera reply to the merchant.\n"
        "Return JSON only: {\"action\":\"send\",\"body\":\"...\",\"cta\":\"binary_yes_no|binary_confirm_cancel|multi_choice_slot|open_ended|none\",\"rationale\":\"...\"}\n\n"
        "Rules:\n"
        "- Use only facts in FACTS.\n"
        "- Do not invent prices, dates, sources, competitors, or completed actions.\n"
        "- Do not ask another qualifying question if the merchant is asking for details.\n"
        "- Answer the merchant's actual reply first, then move to one next step.\n"
        "- Use this shape when sending: direct answer -> grounded reason -> concrete next step.\n"
        "- Keep it under 500 characters and WhatsApp-natural.\n"
        "- Use urgency_guidance to decide how direct the reply should sound.\n"
        "- Follow language_guidance. If Hinglish is requested, use natural Roman Hindi-English mix, not forced translation.\n"
        "- One clear next step in the final sentence.\n"
        "- Do not use Reply YES, Reply CONFIRM, confirm, guaranteed, definitely, or unsupported promises.\n"
        "- Do not use internal labels, IDs, field names, or system words in jargon_policy.\n\n"
        f"FACTS:\n{json.dumps(facts, ensure_ascii=False, indent=2)}"
    )


def _reply_fact_pack(state, merchant_message, mode):
    merchant = state.get("merchant") or {}
    trigger = state.get("trigger") or {}
    customer = state.get("customer") or {}
    category = state.get("category") or {}
    voice = category.get("voice", {}) if category else {}
    return {
        "mode": mode,
        "merchant_message": str(merchant_message)[:500],
        "urgency_guidance": _urgency_guidance(trigger.get("urgency")),
        "language_guidance": _language_guidance(category),
        "jargon_policy": _jargon_policy(),
        "category": {
            "slug": category.get("slug"),
            "tone": voice.get("tone"),
            "code_mix": voice.get("code_mix"),
            "allowed_vocab": voice.get("vocab_allowed", [])[:8],
            "taboo_words": voice.get("vocab_taboo", [])[:8],
        } if category else None,
        "merchant": {
            "name": _short_name(merchant),
            "place": _place(merchant),
            "active_offer": _best_offer(merchant),
            "performance": _performance_facts(merchant),
        },
        "trigger": {
            "kind": trigger.get("kind") or state.get("trigger_kind"),
            "payload": trigger.get("payload", {}),
            "urgency": trigger.get("urgency"),
        },
        "customer": {
            "name": (customer.get("identity") or {}).get("name"),
            "state": customer.get("state"),
        } if customer else None,
        "previous_bot_bodies": list(state.get("sent_bodies", []))[-3:],
        "recent_turns": _recent_turns(state),
        "allowed_ctas": sorted(VALID_REPLY_CTAS),
    }


def _urgency_guidance(urgency):
    try:
        value = int(urgency)
    except (TypeError, ValueError):
        value = 2
    guidance = {
        1: "Light curiosity. No pressure; make it feel optional and useful.",
        2: "Useful FYI. Soft ask; explain relevance without sounding urgent.",
        3: "Timely opportunity. Clear next step; mild urgency is okay.",
        4: "Important action soon. Be direct, concise, and outcome-led.",
        5: "Urgent. Lead with the risk/action first; keep it very concise.",
    }
    return guidance.get(max(1, min(5, value)), guidance[2])


def _language_guidance(category):
    voice = (category or {}).get("voice", {})
    code_mix = str(voice.get("code_mix", "")).lower()
    if "english_primary" in code_mix:
        return "Mostly English. One short Roman Hindi phrase is okay if natural."
    if any(token in code_mix for token in ["hindi", "hinglish", "hi-en", "natural"]):
        return (
            "Use natural Roman Hinglish where it improves warmth: 1-2 short Hindi phrases max. "
            "Keep salutation, names, places, prices, dates, offers, source names, and technical terms in English."
        )
    return "Use clear English in the category voice."


def _jargon_policy():
    return {
        "avoid_words_or_patterns": LOW_SIGNAL_JARGON,
        "safe_replacements": {
            "trigger": "topic or update",
            "payload": "details",
            "ctr_below_peer_median": "profile traffic is not converting as strongly as nearby peers",
            "high_risk_adult_cohort": "this is relevant to your patient mix",
        },
    }


def _parse_json(text):
    if not text or text.startswith("[Claude error:"):
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        value = json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _performance_facts(merchant):
    perf = merchant.get("performance") or {}
    facts = []
    if perf.get("views") is not None:
        facts.append(f"{_number_label(perf['views'])} views")
    if perf.get("calls") is not None:
        facts.append(f"{_number_label(perf['calls'])} calls")
    if perf.get("ctr") is not None:
        facts.append(f"CTR {_pct(perf['ctr'])}")
    delta = perf.get("delta_7d") or {}
    if delta.get("calls_pct") is not None:
        facts.append(f"calls {_pct(delta['calls_pct'], signed=True)} this week")
    return facts[:4]


def _number_label(value):
    try:
        return f"{int(float(value)):,}"
    except (TypeError, ValueError):
        return str(value)


def _recent_turns(state):
    turns = []
    for turn in list(state.get("turns", []))[-5:]:
        speaker = turn.get("from", "unknown")
        message = str(turn.get("message", "")).strip()
        if message:
            turns.append(f"{speaker}: {message[:220]}")
    return turns


def _seen_merchant_reply_count(state, normalized):
    turns = state.get("turns", [])
    count = 0
    for turn in turns:
        if turn.get("from") in {"merchant", "customer"} and normalize_reply(turn.get("message")) == normalized:
            count += 1
    merchant_memory_count = list(state.get("merchant_reply_memory", [])).count(normalized)
    return max(1, count, merchant_memory_count)


def _send_once(state, body, cta, rationale):
    sent = [normalize_reply(item) for item in state.get("sent_bodies", [])]
    if normalize_reply(body) in sent:
        body = _alternate_body(state)
    return {"action": "send", "body": body, "cta": cta, "rationale": rationale}


def _is_customer_scope(state):
    trigger = state.get("trigger") or {}
    last_from = ""
    turns = state.get("turns") or []
    if turns:
        last_from = str(turns[-1].get("from", "")).lower()
    return bool(state.get("customer")) or trigger.get("scope") == "customer" or last_from == "customer"


def _customer_reply(state, text, original_message):
    if is_soft_yes(text) or is_commitment(text):
        return _send_once(state, _customer_confirm_body(state), "binary_yes_no",
                          "Customer accepted the reminder or slot; sending a clear confirmation next step.")
    if any(pattern in text for pattern in CUSTOMER_CHANGE_PATTERNS):
        return _send_once(state, _customer_change_body(state), "open_ended",
                          "Customer needs a change; asking for the concrete slot or prescription change.")
    if is_detail_question(text) or is_price_objection(text):
        return _send_once(state, _customer_detail_body(state), "multi_choice_slot",
                          "Customer asked for detail; answering only from merchant/offer context.")
    llm_decision = _llm_reply(state, original_message, mode="customer")
    if llm_decision:
        return _send_once(state, llm_decision["body"], llm_decision["cta"], llm_decision["rationale"])
    return _send_once(state, _customer_ack_body(state), "multi_choice_slot",
                      "Customer reply was ambiguous; acknowledged and offered one safe next step.")


def _commitment_body(state):
    merchant = state.get("merchant") or {}
    trigger = state.get("trigger") or {}
    kind = trigger.get("kind") or state.get("trigger_kind") or "this"
    business = _short_name(merchant)
    offer = _best_offer(merchant)

    if kind in {"research_digest", "cde_opportunity"}:
        return (
            f"Perfect. I will pull the source note and draft a 4-line WhatsApp for {business}'s customers. "
            "After I share it, you can approve it or ask for one change."
        )
    if kind == "regulation_change":
        return (
            f"Done. I will convert this into a simple checklist for {business}, with only the deadline and action items. "
            "Do you want me to prepare it for your team?"
        )
    if kind in {"perf_dip", "perf_spike", "gbp_unverified"}:
        return (
            f"Great. I will make the profile fix practical: one Google update, one offer line around {offer}, and one metric to watch. "
            "Do you want me to prepare the approval draft?"
        )
    if kind in {"active_planning_intent", "festival_upcoming", "ipl_match_today"}:
        return (
            f"Great, moving to draft mode. I will write the campaign around {offer} and keep it short enough to send today. "
            "Do you want me to queue the draft after review?"
        )
    if kind in {"competitor_opened", "review_theme_emerged", "milestone_reached"}:
        return (
            f"Done. I will draft a proof-led response for {business} so it does not sound like a discount fight. "
            "Do you want me to prepare the version for approval?"
        )
    return (
        f"Great. I will turn this into one clean next step for {business}, grounded in the topic we discussed. "
        "Do you want me to prepare the approval draft?"
    )


def _sample_body(state):
    merchant = state.get("merchant") or {}
    trigger = state.get("trigger") or {}
    topic = _topic_label(trigger)
    proof = _merchant_proof(merchant)
    offer = _best_offer(merchant)
    return (
        f"Sample direction: lead with {topic}, use {proof}, and keep the action around {offer}. "
        "Want me to prepare the full approval draft in that style?"
    )


def _edit_body(state):
    merchant = state.get("merchant") or {}
    return (
        f"Sure. I can edit the draft for {_short_name(merchant)} without changing the facts. "
        "What should I change: shorter, softer tone, different offer, or one line removed?"
    )


def _price_objection_body(state):
    merchant = state.get("merchant") or {}
    offer = _best_offer(merchant)
    return (
        f"Fair point. We do not need a big spend first; the safer move is one small draft around {offer} and your current profile facts. "
        "Want me to make it a low-risk version?"
    )


def _proof_body(state):
    merchant = state.get("merchant") or {}
    proof = _merchant_proof(merchant)
    return (
        f"No guarantee, but the reason to try is grounded: {proof}. "
        "Want me to draft one version you can review before anything goes out?"
    )


def _already_done_body(state):
    merchant = state.get("merchant") or {}
    trigger = state.get("trigger") or {}
    kind = _topic_label(trigger)
    return (
        f"Good, then I will not repeat it. For {_short_name(merchant)}, the useful check is whether {kind} is already visible to customers. "
        "Want me to draft the follow-up version only if there is still a gap?"
    )


def _detail_body(state):
    merchant = state.get("merchant") or {}
    trigger = state.get("trigger") or {}
    kind = str(trigger.get("kind") or state.get("trigger_kind") or "this").replace("_", " ")
    place = _place(merchant)
    offer = _best_offer(merchant)
    return (
        f"Short version: this is about {kind} for {_short_name(merchant)} in {place}. "
        f"The safest draft should use {offer} and avoid any claim not present in your profile data. Want me to write that version?"
    )


def _off_topic_body(state):
    trigger = state.get("trigger") or {}
    kind = str(trigger.get("kind") or state.get("trigger_kind") or "this update").replace("_", " ")
    return (
        f"That part is better handled by your CA or operations team. For Vera, I can help with the {kind} message only: "
        "one draft, no extra claims, ready for your approval."
    )


def _ack_body(state):
    merchant = state.get("merchant") or {}
    offer = _best_offer(merchant)
    return (
        f"Got it. I will keep the next step practical: one short draft using {offer}, based only on your current profile facts. "
        "Do you want me to prepare that version?"
    )


def _alternate_body(state):
    merchant = state.get("merchant") or {}
    return (
        f"Let me make this more direct for {_short_name(merchant)}: I will prepare one approval-ready draft from the same topic, "
        "then wait for your approval before anything goes out."
    )


def _customer_confirm_body(state):
    merchant = state.get("merchant") or {}
    trigger = state.get("trigger") or {}
    slot = _slot_from_trigger(trigger)
    slot_text = f" for {slot}" if slot else ""
    return (
        f"Thanks, {_customer_name(state)}. {_short_name(merchant)} will keep this ready{slot_text}. "
        "Would you like us to share any preparation details?"
    )


def _customer_change_body(state):
    merchant = state.get("merchant") or {}
    trigger = state.get("trigger") or {}
    if trigger.get("kind") == "chronic_refill_due":
        return (
            f"Sure, {_short_name(merchant)} can update this before dispatch. "
            "What changed in the prescription or delivery address?"
        )
    return (
        f"Sure, {_short_name(merchant)} can adjust it. "
        "Which day or time works better for you?"
    )


def _customer_detail_body(state):
    merchant = state.get("merchant") or {}
    offer = _best_offer(merchant)
    return (
        f"{_short_name(merchant)} can help with that. The current option we have is {offer}; I will avoid adding anything not confirmed by the store. "
        "Would you like the next available slot or delivery option?"
    )


def _customer_ack_body(state):
    merchant = state.get("merchant") or {}
    return (
        f"Got it. {_short_name(merchant)} will keep this simple and confirm only the next step. "
        "Would you like a slot, delivery option, or a call-back?"
    )


def _short_name(merchant):
    return (merchant.get("identity") or {}).get("name") or "your business"


def _place(merchant):
    identity = merchant.get("identity") or {}
    locality = identity.get("locality")
    city = identity.get("city")
    if locality and city:
        return f"{locality}, {city}"
    return locality or city or "your locality"


def _best_offer(merchant):
    for offer in merchant.get("offers", []):
        if offer.get("status") == "active" and offer.get("title"):
            return offer["title"]
    return "your strongest current offer"


def _topic_label(trigger):
    trigger = trigger or {}
    payload = trigger.get("payload") or {}
    for key in ["title", "topic", "intent_topic", "metric", "service", "service_due", "molecule"]:
        if payload.get(key):
            return str(payload[key]).replace("_", " ")
    return str(trigger.get("kind") or "this update").replace("_", " ")


def _merchant_proof(merchant):
    perf = _performance_facts(merchant)
    if perf:
        return ", ".join(perf[:2])
    place = _place(merchant)
    return f"the current {place} profile context"


def _customer_name(state):
    customer = state.get("customer") or {}
    identity = customer.get("identity") or {}
    return identity.get("name") or identity.get("first_name") or "there"


def _slot_from_trigger(trigger):
    payload = (trigger or {}).get("payload") or {}
    for key in ["appointment_iso", "date", "due_date"]:
        if payload.get(key):
            return str(payload[key]).replace("T", " ").replace("+05:30", "")
    slots = payload.get("available_slots") or payload.get("next_session_options") or []
    if slots and isinstance(slots, list):
        first = slots[0] or {}
        return first.get("label") or first.get("iso")
    return ""


def _pct(value, signed=False):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    pct = number * 100 if abs(number) <= 1 else number
    prefix = "+" if signed and pct > 0 else ""
    text = f"{pct:.1f}".rstrip("0").rstrip(".")
    return f"{prefix}{text}%"
