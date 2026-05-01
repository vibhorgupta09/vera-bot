import os
import re
import time
import uuid

# Load .env BEFORE any other imports that need environment variables
root_dir = os.path.dirname(__file__)
env_file = os.path.join(root_dir, ".env")
if os.path.exists(env_file):
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value

# Now import modules that depend on environment variables
from flask import Flask, request, jsonify
from stores.context_store import ContextStore
from stores.category_store import CategoryStore
from triggers.trigger_manager import TriggerManager
from messaging.composer import Composer
from messaging.conversation_handlers import respond as respond_to_reply
from messaging.validator import Validator

app = Flask(__name__)
start_time = time.time()

store = ContextStore()
category_store = CategoryStore(os.path.join(root_dir, "dataset", "categories"))
trigger_manager = TriggerManager(store)
composer = Composer(category_store)
validator = Validator()
conversations = {}
reply_memory = {}
conversation_index = {}
sent_suppression_keys = set()

TEAM_NAME = "Vera Challenge Bot"
TEAM_MEMBERS = ["Developer"]
CONTACT_EMAIL = "team@example.com"
VERSION = "0.1.0"

VALID_SCOPES = {"category", "merchant", "customer", "trigger"}


@app.route("/v1/healthz", methods=["GET"])
def healthz():
    return jsonify({
        "status": "ok",
        "uptime_seconds": int(time.time() - start_time),
        "contexts_loaded": store.counts(),
    })


@app.route("/v1/metadata", methods=["GET"])
def metadata():
    return jsonify({
        "team_name": TEAM_NAME,
        "team_members": TEAM_MEMBERS,
        "model": composer.model_name,
        "approach": "4-context composer with trigger selection and category-aware validation",
        "contact_email": CONTACT_EMAIL,
        "version": VERSION,
        "submitted_at": "2026-04-30T00:00:00Z",
    })


@app.route("/v1/context", methods=["POST"])
def receive_context():
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"accepted": False, "reason": "invalid_json", "details": "Request body must be valid JSON."}), 400

    scope = body.get("scope")
    context_id = body.get("context_id")
    version = body.get("version")
    payload = body.get("payload")

    if scope not in VALID_SCOPES:
        return jsonify({"accepted": False, "reason": "invalid_scope", "details": f"scope must be one of {sorted(VALID_SCOPES)}."}), 400
    if not context_id or not isinstance(context_id, str):
        return jsonify({"accepted": False, "reason": "invalid_context_id", "details": "context_id must be a non-empty string."}), 400
    if not isinstance(version, int):
        return jsonify({"accepted": False, "reason": "invalid_version", "details": "version must be an integer."}), 400
    if payload is None:
        return jsonify({"accepted": False, "reason": "invalid_payload", "details": "payload is required."}), 400

    accepted, details = store.add_context(scope, context_id, version, payload)
    if not accepted:
        return jsonify({"accepted": False, **details}), 409

    return jsonify({
        "accepted": True,
        "ack_id": f"ack_{uuid.uuid4().hex[:12]}",
        "stored_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })


@app.route("/v1/tick", methods=["POST"])
def tick():
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"actions": [], "errors": ["Request body must be valid JSON."]}), 400

    now = body.get("now")
    available_triggers = body.get("available_triggers", [])
    if not isinstance(available_triggers, list):
        return jsonify({"actions": [], "errors": ["available_triggers must be a list."]}), 400

    actions = []
    errors = []
    tick_started = time.perf_counter()
    # The judge may pass batches of triggers; cap sends so LLM refinement stays
    # inside the 15s tick timeout while still choosing the top-ranked items.
    try:
        max_actions = int(os.environ.get("VERA_MAX_ACTIONS_PER_TICK", "2"))
    except ValueError:
        max_actions = 2
    max_actions = max(1, max_actions)

    try:
        soft_budget_ms = int(os.environ.get("VERA_TICK_SOFT_BUDGET_MS", "11500"))
    except ValueError:
        soft_budget_ms = 11500
    soft_budget_ms = max(1000, soft_budget_ms)

    for trigger_id in trigger_manager.rank_triggers(available_triggers):
        if len(actions) >= max_actions:
            break

        elapsed_ms = int((time.perf_counter() - tick_started) * 1000)
        if actions and elapsed_ms >= soft_budget_ms:
            print(
                f"[TICK] stopping after {len(actions)} action(s): "
                f"elapsed {elapsed_ms}ms reached soft budget {soft_budget_ms}ms",
                flush=True,
            )
            break

        trigger = store.get_context("trigger", trigger_id)
        if not trigger:
            continue
        suppression_key = trigger.get("suppression_key") or trigger.get("id")
        if suppression_key in sent_suppression_keys:
            continue

        merchant_id = trigger.get("merchant_id")
        customer_id = trigger.get("customer_id")
        merchant = store.get_context("merchant", merchant_id) if merchant_id else None
        if not merchant:
            errors.append(f"{trigger_id}: missing merchant context")
            continue

        customer = store.get_context("customer", customer_id) if customer_id else None
        if trigger.get("scope") == "customer" and customer_id and not customer:
            errors.append(f"{trigger_id}: missing customer context")
            continue

        category = resolve_category(merchant, trigger)
        compose_started = time.perf_counter()
        action, rationale = composer.compose(category, merchant, trigger, customer)
        compose_ms = int((time.perf_counter() - compose_started) * 1000)
        elapsed_ms = int((time.perf_counter() - tick_started) * 1000)
        print(
            f"[TICK] composed {trigger_id} in {compose_ms}ms "
            f"(tick elapsed {elapsed_ms}ms)",
            flush=True,
        )

        valid, problems = validator.validate(action, category, merchant, trigger, customer)
        if not valid:
            errors.append(f"{trigger_id}: {'; '.join(problems)}")
            continue

        action["rationale"] = rationale
        actions.append(action)
        record_outbound(action, category, merchant, trigger, customer)
        if action.get("suppression_key"):
            sent_suppression_keys.add(action["suppression_key"])

        elapsed_ms = int((time.perf_counter() - tick_started) * 1000)
        if elapsed_ms >= soft_budget_ms:
            print(
                f"[TICK] returning {len(actions)} action(s): "
                f"elapsed {elapsed_ms}ms reached soft budget {soft_budget_ms}ms",
                flush=True,
            )
            break

    response = {"actions": actions}
    if errors and not actions:
        response["errors"] = errors
    return jsonify(response)


@app.route("/v1/reply", methods=["POST"])
def reply():
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"action": "end", "rationale": "invalid JSON reply payload"}), 400

    conversation_id = body.get("conversation_id")
    merchant_id = body.get("merchant_id")
    customer_id = body.get("customer_id")
    message = body.get("message", "") or ""
    if not conversation_id or not isinstance(message, str):
        return jsonify({"action": "end", "rationale": "missing conversation_id or message"}), 400

    turn = {
        "from": body.get("from_role", "merchant"),
        "message": message,
        "received_at": body.get("received_at"),
        "turn_number": body.get("turn_number"),
        "merchant_id": merchant_id,
        "customer_id": customer_id,
    }
    conversations.setdefault(conversation_id, []).append(turn)
    normalized = normalize_reply(message)
    if merchant_id:
        reply_memory.setdefault(merchant_id, []).append(normalized)
        reply_memory[merchant_id] = reply_memory[merchant_id][-8:]

    state = conversation_index.setdefault(conversation_id, {
        "conversation_id": conversation_id,
        "merchant_id": merchant_id,
        "customer_id": customer_id,
        "sent_bodies": [],
        "turns": [],
    })
    if merchant_id and not state.get("merchant_id"):
        state["merchant_id"] = merchant_id
    if customer_id and not state.get("customer_id"):
        state["customer_id"] = customer_id
    state["turns"] = conversations.get(conversation_id, [])
    state["merchant_reply_memory"] = reply_memory.get(state.get("merchant_id"), [])
    state["merchant"] = store.get_context("merchant", state.get("merchant_id")) if state.get("merchant_id") else None
    state["customer"] = store.get_context("customer", state.get("customer_id")) if state.get("customer_id") else None
    if state.get("trigger_id"):
        state["trigger"] = store.get_context("trigger", state["trigger_id"])
    if not state.get("category"):
        state["category"] = resolve_category(state.get("merchant"), state.get("trigger"))

    decision = respond_to_reply(state, message)
    valid, problems = validator.validate_reply(decision, state)
    if not valid:
        return jsonify({
            "action": "end",
            "rationale": "Reply validation failed: " + "; ".join(problems),
        })

    if decision.get("action") == "send":
        state.setdefault("sent_bodies", []).append(decision.get("body", ""))
        conversations.setdefault(conversation_id, []).append({
            "from": "vera",
            "message": decision.get("body", ""),
            "received_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "turn_number": body.get("turn_number"),
            "merchant_id": merchant_id,
            "customer_id": customer_id,
        })
    if decision.get("action") == "end":
        state["status"] = "ended"

    return jsonify(decision)


@app.route("/v1/teardown", methods=["POST"])
def teardown():
    store.clear()
    conversations.clear()
    reply_memory.clear()
    conversation_index.clear()
    sent_suppression_keys.clear()
    return jsonify({"accepted": True, "cleared": True})


def resolve_category(merchant, trigger=None):
    slug = None
    if merchant:
        slug = merchant.get("category_slug") or merchant.get("category")
    if not slug and trigger:
        slug = trigger.get("payload", {}).get("category")
    if slug:
        pushed_category = store.get_context("category", slug)
        if pushed_category:
            return pushed_category
        local_category = category_store.get_category(slug)
        if local_category:
            return local_category
    return None


def record_outbound(action, category, merchant, trigger, customer=None):
    conversation_id = action.get("conversation_id")
    if not conversation_id:
        return
    state = conversation_index.setdefault(conversation_id, {
        "conversation_id": conversation_id,
        "sent_bodies": [],
        "turns": [],
    })
    state.update({
        "merchant_id": action.get("merchant_id"),
        "customer_id": action.get("customer_id"),
        "trigger_id": action.get("trigger_id"),
        "trigger_kind": (trigger or {}).get("kind"),
        "category_slug": (category or {}).get("slug"),
        "category": category,
        "merchant": merchant,
        "customer": customer,
        "trigger": trigger,
        "status": "open",
    })
    state.setdefault("sent_bodies", []).append(action.get("body", ""))
    turn = {
        "from": "vera" if action.get("send_as") == "vera" else "merchant_on_behalf",
        "message": action.get("body", ""),
        "sent_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "trigger_id": action.get("trigger_id"),
        "merchant_id": action.get("merchant_id"),
        "customer_id": action.get("customer_id"),
    }
    conversations.setdefault(conversation_id, []).append(turn)
    state["turns"] = conversations[conversation_id]


def normalize_reply(message):
    return re.sub(r"\s+", " ", message.strip().lower())


def is_auto_reply(text):
    patterns = [
        "thank you for contacting",
        "thanks for contacting",
        "our team will respond",
        "will respond shortly",
        "automated assistant",
        "auto-reply",
        "business hours",
        "we will get back",
    ]
    return any(pattern in text for pattern in patterns)


def is_hostile_or_optout(text):
    patterns = [
        "stop messaging",
        "stop sending",
        "not interested",
        "unsubscribe",
        "do not message",
        "dont message",
        "don't message",
        "useless spam",
        "spam",
    ]
    return any(pattern in text for pattern in patterns)


def is_commitment(text):
    patterns = [
        "lets do it",
        "let's do it",
        "go ahead",
        "proceed",
        "confirm",
        "do it",
        "what's next",
        "whats next",
        "start it",
    ]
    return any(pattern in text for pattern in patterns)


def is_soft_yes(text):
    return text in {"yes", "yes please", "ok", "okay", "sure"} or text.startswith("yes ")


def is_off_topic(text):
    off_topic_terms = ["gst", "tax filing", "income tax", "itr", "loan", "accounting"]
    return any(term in text for term in off_topic_terms)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
