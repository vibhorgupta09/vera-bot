# Vera Challenge Bot

## Approach

This implementation uses a hybrid appraoch by using LLMs and also template data incase of fallbacks. 

### Approach: Hybrid Message Flow

1. Build a deterministic draft first, then call the LLM with a prompt chosen by trigger class. Merchant triggers are grouped into 5 prompt profiles: evidence, performance, campaign, reputation, and winback, with a generic profile for unknown triggers.
   - This call receives a grounded fact pack with category voice, merchant performance, offers, trigger payload, urgency, CTA instruction, and the baseline draft. It rewrites the message for relevance and specificity without inventing new facts.

2. Call the LLM a second time for polish.
   - This pass is narrower: it improves WhatsApp tone, light code-mix, CTA phrasing, and readability while preserving facts, names, dates, prices, sources, and the original message structure.

3. Validate the LLM output.
   - Validation checks required schema fields, send_as, CTA, empty body, grounding, taboo words, unsupported claims, low-signal jargon leaks, source hallucinations, and whether the final merchant CTA is a concrete question.

4. If the LLM message is rejected, send the deterministic template.
   - The template is built only from the provided context, so it remains a safe fallback when the LLM is slow, unavailable, malformed, too generic, or fails validation.

### Approach: Hybrid Reply Flow

1. Run a deterministic safety gate before using the LLM.
   - This catches hostile replies, opt-outs, wrong numbers, repeated auto-replies, and obvious pause requests. These cases route directly to `end` or `wait` so the bot does not over-message.

2. Call the LLM as an intent classifier for normal or ambiguous replies.
   - This call does not draft a message. It only maps the incoming WhatsApp reply to fixed keys such as commitment, sample_request, edit_request, price_objection, proof_question, already_done, off_topic, customer_confirm, customer_change, customer_detail, or ambiguous.

3. Route the classified intent to deterministic reply handlers.
   - Each handler produces a controlled response for the next turn, such as moving to action mode after commitment, answering a proof question, handling a price objection, asking for an edit preference, confirming a customer slot, or asking for a different customer time.

4. Validate the reply response.
   - Validation checks the reply action schema, non-empty body, CTA label, URL/boilerplate leakage, low-signal jargon, repeated body text, and valid wait_seconds for wait actions.

5. If classification fails, fall back to rule-based routing.
   - This keeps `/reply` reliable when the LLM is unavailable, uncertain, malformed, or too slow.




`bot.py` exposes the standalone `compose(...)` function requested in the brief. `app.py` exposes the live HTTP endpoints and keeps per-conversation memory for suppression keys, sent bodies, and auto-reply detection. Set `VERA_USE_LLM=0` to force deterministic-only mode.



## Project Layout

Runtime code is grouped by responsibility:

```text
app.py                      # Flask API entrypoint and judge contract
bot.py                      # standalone compose(...) helper
messaging/composer.py       # outbound /tick message composition
messaging/conversation_handlers.py  # /reply routing and response logic
messaging/validator.py      # schema, grounding, and safety validation
stores/                     # category/context stores
triggers/                   # trigger ranking
llm/                        # Anthropic client wrapper
dataset/categories/         # category knowledge files
```

NOTE : Root modules such as `composer.py`, `validator.py`, and `llm_client.py` are compatibility wrappers so older scripts and deployment commands keep working.
