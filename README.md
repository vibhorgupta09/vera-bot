# Vera Challenge Bot

## Approach

This implementation keeps the Flask judge contract intact and focuses the bot on grounded WhatsApp composition. The composer dispatches by `trigger.kind` and uses only the provided category, merchant, trigger, and customer contexts. Category voice, taboo vocabulary, offer catalog, peer stats, digest items, trend signals, merchant performance, active offers, and customer preferences are folded into deterministic baseline templates.

When `ANTHROPIC_API_KEY` is available, merchant-facing baseline messages are refined once through Claude using a compact fact pack. The refined body is accepted only if validation passes; otherwise the deterministic baseline is returned. Reply handling still uses rules for safety-critical cases such as opt-out, auto-replies, and clear commitment; Claude is used only for ambiguous/detail replies with the same fallback posture.

`bot.py` exposes the standalone `compose(...)` function requested in the brief. `conversation_handlers.py` contains the optional replay/tiebreaker reply handler. `app.py` exposes the live HTTP endpoints and keeps per-conversation memory for suppression keys, sent bodies, and auto-reply detection.

The main tradeoff is controlled creativity: the LLM can improve naturalness and engagement, but the service still has a safe fallback for API failure, malformed JSON, taboo words, or unsupported claims. Set `VERA_USE_LLM=0` to force deterministic-only mode.

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the API:

```bash
python3 app.py
```

Generate the offline submission:

```bash
python3 generate_submission.py
```

The script writes `submission.jsonl` with 30 validated rows when enough valid contexts are available.

Run the official local judge simulator:

```bash
python3 judge_simulator.py
```

Before running it, set `BOT_URL`, `LLM_PROVIDER`, `LLM_API_KEY`, and optionally `LLM_MODEL` in the configuration block at the top of `judge_simulator.py`.

## Useful Checks

```bash
python3 -m py_compile app.py bot.py composer.py validator.py generate_submission.py conversation_handlers.py judge_simulator.py
python3 generate_submission.py
python3 judge_simulator.py
```

If port `8080` is already in use, set `PORT` before starting the Flask app.
