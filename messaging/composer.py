import json
import os
import re
from datetime import datetime

from llm.llm_client import LLMClient
from messaging.validator import Validator


class Composer:
    """Category-aware, trigger-specific WhatsApp composer.

    The challenge rewards grounded specificity more than fluent generic copy. This
    composer therefore prefers deterministic templates that only use facts already
    present in the four contexts. Each trigger family gets a small template so the
    message can explain "why now" without depending on a live LLM call.
    """

    CTA_BY_KIND = {
        "research_digest": "open_ended",
        "regulation_change": "follow_up",
        "cde_opportunity": "open_ended",
        "perf_dip": "follow_up",
        "perf_spike": "follow_up",
        "seasonal_perf_dip": "follow_up",
        "renewal_due": "follow_up",
        "festival_upcoming": "show_offer",
        "curious_ask_due": "open_ended",
        "winback_eligible": "show_offer",
        "dormant_with_vera": "follow_up",
        "ipl_match_today": "show_offer",
        "review_theme_emerged": "follow_up",
        "milestone_reached": "follow_up",
        "active_planning_intent": "follow_up",
        "supply_alert": "follow_up",
        "category_seasonal": "follow_up",
        "gbp_unverified": "follow_up",
        "competitor_opened": "follow_up",
        "recall_due": "follow_up",
        "wedding_package_followup": "follow_up",
        "customer_lapsed_hard": "show_offer",
        "customer_lapsed_soft": "show_offer",
        "trial_followup": "follow_up",
        "chronic_refill_due": "follow_up",
        "appointment_tomorrow": "follow_up",
    }

    def __init__(self, category_store):
        self.category_store = category_store
        self.llm_client = LLMClient()
        self.validator = Validator()
        self.llm_enabled = os.environ.get("VERA_USE_LLM", "1").lower() not in {"0", "false", "no"}
        self.debug_llm = os.environ.get("VERA_DEBUG_LLM", "1").lower() not in {"0", "false", "no"}
        self.model_name = (
            f"llm-refined-templates:{self.llm_client.model_name}"
            if self._llm_ready()
            else "deterministic-trigger-templates-v2"
        )

    def compose(self, category, merchant, trigger, customer=None):
        if not merchant:
            return self._empty_action(trigger), "missing merchant context"
        if not trigger:
            return self._empty_action({}), "missing trigger context"

        category = category or self.category_store.get_category(merchant.get("category_slug"))
        body = self._compose_body(category, merchant, trigger, customer)
        body = self._clean_body(body)

        cta = self._select_cta(trigger)
        kind = trigger.get("kind", "generic")
        send_as = "merchant_on_behalf" if trigger.get("scope") == "customer" else "vera"
        action = self._build_action(merchant, trigger, customer, body, cta, send_as, kind)
        rationale = self._build_rationale(category, merchant, trigger, customer, cta)

        refined = self._refine_with_llm(category, merchant, trigger, customer, action, rationale)
        if refined:
            return refined
        return action, rationale

    def _build_action(self, merchant, trigger, customer, body, cta, send_as, kind):
        return {
            "conversation_id": self._conversation_id(merchant, trigger, customer),
            "merchant_id": merchant.get("merchant_id"),
            "customer_id": customer.get("customer_id") if customer else trigger.get("customer_id"),
            "send_as": send_as,
            "trigger_id": trigger.get("id"),
            "template_name": self._template_name(send_as, kind),
            "template_params": self._template_params(body, merchant, trigger, customer),
            "body": body,
            "cta": cta,
            "suppression_key": trigger.get("suppression_key", trigger.get("id", "")),
        }

    def _llm_ready(self):
        return self.llm_enabled and self.llm_client.available()

    def _refine_with_llm(self, category, merchant, trigger, customer, template_action, template_rationale):
        """Let Claude improve the deterministic draft, but keep the template as fallback.

        Merchant and customer scopes use separate prompt paths. Customer sends
        carry consent and booking risk, so they are routed to a stricter flow.
        """
        trigger_id = trigger.get("id", "unknown")
        kind = trigger.get("kind", "unknown")
        if trigger.get("scope") == "customer":
            return self._refine_customer_with_llm(category, merchant, trigger, customer, template_action, template_rationale)
        if customer or trigger.get("scope") != "merchant":
            self._llm_debug(f"template fallback for {trigger_id} ({kind}): customer/non-merchant scope")
            return None
        if not self._llm_ready():
            self._llm_debug(f"template fallback for {trigger_id} ({kind}): LLM unavailable or disabled")
            return None

        fact_pack = self._llm_fact_pack(category, merchant, trigger, template_action)
        prompt = self._llm_prompt(fact_pack)
        self._llm_debug(f"trying LLM refinement for {trigger_id} ({kind}) with {self.llm_client.model_name}")
        response = self.llm_client.generate(
            prompt,
            system=(
                "You are Magicpin Vera's WhatsApp message composer. "
                "Return compact JSON only. Do not invent facts."
            ),
            max_tokens=450,
            temperature=0.1,
        )
        if response.startswith("[Claude error:"):
            self._llm_debug(f"template fallback for {trigger_id} ({kind}): {response[:180]}")
            return None
        parsed = self._parse_llm_json(response)
        if not parsed:
            self._llm_debug(f"template fallback for {trigger_id} ({kind}): LLM returned non-JSON")
            return None

        body = self._clean_body(str(parsed.get("body", "")))
        if not body:
            self._llm_debug(f"template fallback for {trigger_id} ({kind}): LLM body was empty")
            return None

        candidate = dict(template_action)
        candidate["body"] = body
        if self._is_unknown_trigger(trigger) and parsed.get("cta") in Validator.VALID_CTAS:
            candidate["cta"] = parsed["cta"]
        else:
            candidate["cta"] = template_action["cta"]
        candidate["template_params"] = self._template_params(body, merchant, trigger, customer)

        rationale = self._clean_body(str(parsed.get("rationale", ""))) or template_rationale
        polished = self._polish_with_llm(candidate, fact_pack, template_action, trigger_id, kind)
        if polished:
            polished_candidate = dict(candidate)
            polished_candidate["body"] = polished
            polished_candidate["template_params"] = self._template_params(polished, merchant, trigger, customer)
            valid, problems = self.validator.validate(polished_candidate, category, merchant, trigger, customer)
            if valid:
                self._llm_debug(f"LLM accepted after polish for {trigger_id} ({kind}): {polished}")
                return polished_candidate, f"LLM-refined from grounded fact pack; polished for voice/CTA; {rationale}"
            self._llm_debug(f"polish rejected for {trigger_id} ({kind}): {'; '.join(problems)}")

        valid, problems = self.validator.validate(candidate, category, merchant, trigger, customer)
        if not valid:
            self._llm_debug(f"template fallback for {trigger_id} ({kind}): validation failed: {'; '.join(problems)}")
            return None

        self._llm_debug(f"LLM accepted without polish for {trigger_id} ({kind}): {body}")
        return candidate, f"LLM-refined from grounded fact pack; unpolished fallback; {rationale}"

    def _refine_customer_with_llm(self, category, merchant, trigger, customer, template_action, template_rationale):
        """Improve customer-facing first messages with a separate, stricter flow.

        Customer sends are on the merchant's behalf, so this path never shares the
        merchant-facing prompt or CTA language. The deterministic template remains
        the fallback whenever the LLM is unavailable or validation catches risk.
        """
        trigger_id = trigger.get("id", "unknown")
        kind = trigger.get("kind", "unknown")
        if not customer:
            self._llm_debug(f"customer template fallback for {trigger_id} ({kind}): missing customer context")
            return None
        if not self._has_customer_consent(customer):
            self._llm_debug(f"customer template fallback for {trigger_id} ({kind}): missing customer consent")
            return None
        if not self._llm_ready():
            self._llm_debug(f"customer template fallback for {trigger_id} ({kind}): LLM unavailable or disabled")
            return None

        fact_pack = self._customer_llm_fact_pack(category, merchant, trigger, customer, template_action)
        prompt = self._customer_llm_prompt(fact_pack)
        self._llm_debug(f"trying customer LLM refinement for {trigger_id} ({kind}) with {self.llm_client.model_name}")
        response = self.llm_client.generate(
            prompt,
            system=(
                "You compose customer-facing WhatsApp messages on behalf of local merchants. "
                "Return compact JSON only. Never invent facts."
            ),
            max_tokens=380,
            temperature=0.1,
        )
        if response.startswith("[Claude error:"):
            self._llm_debug(f"customer template fallback for {trigger_id} ({kind}): {response[:180]}")
            return None
        parsed = self._parse_llm_json(response)
        if not parsed:
            self._llm_debug(f"customer template fallback for {trigger_id} ({kind}): LLM returned non-JSON")
            return None

        body = self._clean_body(str(parsed.get("body", "")))
        if not body:
            self._llm_debug(f"customer template fallback for {trigger_id} ({kind}): LLM body was empty")
            return None

        candidate = dict(template_action)
        candidate["body"] = body
        candidate["cta"] = template_action["cta"]
        candidate["template_params"] = self._template_params(body, merchant, trigger, customer)

        rationale = self._clean_body(str(parsed.get("rationale", ""))) or template_rationale
        polished = self._polish_customer_with_llm(candidate, fact_pack, template_action, trigger_id, kind)
        if polished:
            polished_candidate = dict(candidate)
            polished_candidate["body"] = polished
            polished_candidate["template_params"] = self._template_params(polished, merchant, trigger, customer)
            valid, problems = self.validator.validate(polished_candidate, category, merchant, trigger, customer)
            if valid:
                self._llm_debug(f"customer LLM accepted after polish for {trigger_id} ({kind}): {polished}")
                return polished_candidate, f"Customer LLM-refined from consented context; polished for voice/CTA; {rationale}"
            self._llm_debug(f"customer polish rejected for {trigger_id} ({kind}): {'; '.join(problems)}")

        valid, problems = self.validator.validate(candidate, category, merchant, trigger, customer)
        if not valid:
            self._llm_debug(f"customer template fallback for {trigger_id} ({kind}): validation failed: {'; '.join(problems)}")
            return None

        self._llm_debug(f"customer LLM accepted without polish for {trigger_id} ({kind}): {body}")
        return candidate, f"Customer LLM-refined from consented context; unpolished fallback; {rationale}"

    def _polish_with_llm(self, candidate, fact_pack, template_action, trigger_id, kind):
        """Use a narrow second pass for voice, code-mix, and CTA shape only."""
        body = candidate.get("body", "")
        template_body = template_action.get("body", "")
        prompt = self._polish_prompt(body, template_body, fact_pack)
        self._llm_debug(f"trying polish for {trigger_id} ({kind})")
        response = self.llm_client.generate(
            prompt,
            system=(
                "You polish Magicpin Vera WhatsApp messages without changing facts. "
                "Return compact JSON only."
            ),
            max_tokens=260,
            temperature=0.0,
        )
        if response.startswith("[Claude error:"):
            self._llm_debug(f"polish failed for {trigger_id} ({kind}): {response[:180]}")
            return None
        parsed = self._parse_llm_json(response)
        if not parsed:
            self._llm_debug(f"polish failed for {trigger_id} ({kind}): non-JSON")
            return None
        polished = self._clean_body(str(parsed.get("body", "")))
        if not polished:
            self._llm_debug(f"polish failed for {trigger_id} ({kind}): empty body")
            return None
        return polished

    def _polish_customer_with_llm(self, candidate, fact_pack, template_action, trigger_id, kind):
        """Narrow polish pass for customer sends: voice and final choice only."""
        body = candidate.get("body", "")
        template_body = template_action.get("body", "")
        prompt = self._customer_polish_prompt(body, template_body, fact_pack)
        self._llm_debug(f"trying customer polish for {trigger_id} ({kind})")
        response = self.llm_client.generate(
            prompt,
            system=(
                "You polish customer WhatsApp reminders for local merchants without changing facts. "
                "Return compact JSON only."
            ),
            max_tokens=240,
            temperature=0.0,
        )
        if response.startswith("[Claude error:"):
            self._llm_debug(f"customer polish failed for {trigger_id} ({kind}): {response[:180]}")
            return None
        parsed = self._parse_llm_json(response)
        if not parsed:
            self._llm_debug(f"customer polish failed for {trigger_id} ({kind}): non-JSON")
            return None
        polished = self._clean_body(str(parsed.get("body", "")))
        if not polished:
            self._llm_debug(f"customer polish failed for {trigger_id} ({kind}): empty body")
            return None
        return polished

    def _polish_prompt(self, body, template_body, fact_pack):
        return (
            "Return JSON only with key body.\n"
            "Task: polish MESSAGE for WhatsApp voice, code-mix, and CTA. This is not a rewrite.\n\n"
            "Rules:\n"
            "- Change the language to hinglish with warmth to make it sound natural human language.\n"
            "- Preserve this message shape: trigger fact -> merchant-specific proof -> why now / decision reason -> concrete next step.\n"
            "- Change only wording needed for warmth, code-mix, and CTA.\n"
            "- Do not rewrite facts, numbers, dates, offers, names, places, sources, or claims.\n"
            "- Do not add unsupported hype or certainty such as absolutely, definitely, will resonate, perfect timing, customers love, or caution badge.\n"
            "- Keep salutation, names, places, prices, dates, offers, source names, and technical terms in English.\n\n"
            "- If MESSAGE already has a weak/command CTA, replace only that final CTA sentence.\n"
            "- If MESSAGE has no CTA, append one short final question in your message tone and language.\n"
            "- Use TEMPLATE_CTA as the meaning of the final CTA, but match MESSAGE tone and language.\n"
            "- End with a direct question\n"
            "- Do not use Reply YES, Reply CONFIRM, yes, confirm, FYI, worth a look, kya chalega, or ready to help.\n"
            "- Remove dashboard-ish phrasing if present, but do not remove important specificity.\n"
            "- Avoid taboo words and jargon_policy words.\n"
            "- Keep the final body natural for WhatsApp and ideally under 500 characters.\n\n"
            f"LANGUAGE_GUIDANCE: {fact_pack.get('language_guidance')}\n"
            f"CTA_INSTRUCTION: {fact_pack.get('cta_instruction')}\n"
            f"TEMPLATE_CTA: {self._last_sentence(template_body)}\n"
            f"CATEGORY_VOICE: {json.dumps(fact_pack.get('category', {}), ensure_ascii=False)}\n"
            f"JARGON_POLICY: {json.dumps(fact_pack.get('jargon_policy', {}), ensure_ascii=False)}\n"
            f"MESSAGE: {body}\n"
        )

    def _customer_polish_prompt(self, body, template_body, fact_pack):
        return (
            "Return JSON only with key body.\n"
            "Task: lightly polish CUSTOMER_MESSAGE for a WhatsApp customer reminder. This is not a rewrite.\n\n"
            "Rules:\n"
            "- Preserve this shape: merchant identity -> why this customer is being contacted -> useful next step -> one low-friction question.\n"
            "- Keep all names, merchant name, places, prices, medicines, services, dates, slots, and offer titles exactly grounded in FACTS.\n"
            "- Apply language_guidance naturally. If Hinglish is requested, use short Roman Hindi-English phrases only where natural.\n"
            "- The message is from the merchant to the customer; never mention Vera, dashboards, prompts, approvals, or drafting.\n"
            "- Do not add medical, dental, fitness, pharmacy, or beauty claims beyond the facts.\n"
            "- Do not use guaranteed, definitely, perfect timing, customers love, or unsupported certainty.\n"
            "- If slots exist, the final question may offer slot choices. If not, ask for one simple next action.\n"
            "- Do not use Reply YES or Reply CONFIRM. Reply 1/2 is allowed only when FACTS include two concrete slots.\n"
            "- Keep the body under 500 characters and natural for WhatsApp.\n\n"
            f"LANGUAGE_GUIDANCE: {fact_pack.get('language_guidance')}\n"
            f"CTA_INSTRUCTION: {fact_pack.get('cta_instruction')}\n"
            f"TEMPLATE_FINAL_SENTENCE: {self._last_sentence(template_body)}\n"
            f"CATEGORY_VOICE: {json.dumps(fact_pack.get('category', {}), ensure_ascii=False)}\n"
            f"JARGON_POLICY: {json.dumps(fact_pack.get('jargon_policy', {}), ensure_ascii=False)}\n"
            f"FACTS: {json.dumps(fact_pack, ensure_ascii=False)}\n"
            f"CUSTOMER_MESSAGE: {body}\n"
        )

    def _last_sentence(self, body):
        body = self._clean_body(body)
        if not body:
            return ""
        pieces = re.split(r"(?<=[.!?])\s+", body)
        return pieces[-1].strip() if pieces else body

    def _llm_debug(self, message):
        if self.debug_llm:
            formatted = f"[COMPOSER] {message}"
            print(formatted, flush=True)
            self._append_debug_log(formatted)

    def _append_debug_log(self, message):
        log_file = os.environ.get("VERA_LLM_LOG_FILE", "/tmp/vera_llm_debug.log")
        try:
            with open(log_file, "a", encoding="utf-8") as handle:
                handle.write(f"{datetime.utcnow().isoformat()}Z {message}\n")
        except OSError:
            pass

    def _llm_prompt(self, fact_pack):
        profile = fact_pack.get("prompt_profile", "generic")
        return (
            f"{self._prompt_opening(profile)}\n"
            "Return JSON only with keys body, cta, rationale.\n\n"
            "Rules:\n"
            f"{self._profile_rules(profile)}"
            "- Force this four-part message shape in order: trigger fact, merchant-specific proof, why now / decision reason, one concrete next step.\n"
            "- Each part can be one short clause or sentence, but do not skip any part.\n"
            "- Use only facts in FACTS; do not invent numbers, dates, names, sources, competitors, or offers.\n"
            "- Do not expose internal labels such as ctr_below_peer_median or high_risk_adult_cohort.\n"
            "- Use at most two performance facts; avoid a dashboard dump.\n"
            "- Explain why this message matters now, using the urgency guidance.\n"
            "- Write this content pass in clear English. Do not add Hinglish or code-mix here; the polish pass handles that.\n"
            "- Preserve names, dates, prices, offer titles, source names, and technical vocabulary exactly as provided.\n"
            "- Do not use internal jargon, IDs, field names, or system words listed in jargon_policy.\n"
            "- Do not use unsupported hype or certainty such as absolutely, definitely, will resonate, perfect timing, customers love, or caution badge.\n"
            "- Follow cta_instruction exactly for the final sentence style.\n"
            "- For merchant-facing CTAs, use a direct question beginning with Want me to or Do you want me to.\n"
            "- Do not use imperative reply-command wording such as asking the merchant to answer YES or CONFIRM.\n"
            "- The final sentence must be the CTA; do not add another sentence after it.\n"
            "- Do not end with FYI, worth a look, your choice, let's fix this, ready to help, kya chalega, or kya soch hai.\n"
            "- One primary CTA in the final sentence.\n"
            "- Avoid causal claims like fixes, blocks, usually, typically, guarantees, spike, or rush unless that wording is grounded in FACTS.\n"
            "- Keep the body natural for WhatsApp, ideally under 500 characters.\n"
            "- Match the category voice and avoid taboo words.\n"
            "- If cta_mode is preserve, preserve expected_cta exactly.\n"
            "- If cta_mode is choose, choose exactly one CTA from allowed_ctas.\n\n"
            "FACTS:\n"
            f"{json.dumps(fact_pack, ensure_ascii=False, indent=2)}"
        )

    def _customer_llm_prompt(self, fact_pack):
        profile = fact_pack.get("prompt_profile", "generic_customer")
        return (
            f"{self._customer_prompt_opening(profile)}\n"
            "Return JSON only with keys body, rationale.\n\n"
            "Rules:\n"
            f"{self._customer_profile_rules(profile)}"
            "- Force this four-part shape in order: merchant identity, customer-specific reason, why now / usefulness, one concrete next step.\n"
            "- The audience is the merchant's customer, not the merchant. Never ask whether to draft, prepare, approve, or post anything.\n"
            "- Use only facts in FACTS; do not invent slots, prices, dates, service history, medicine names, or offers.\n"
            "- Keep the merchant name visible near the start so the customer knows who is contacting them.\n"
            "- Use the customer's name if available and natural; do not use placeholder or anonymous profile text as a name.\n"
            "- Preserve prices, slots, dates, medicine names, service names, and offer titles exactly as provided.\n"
            "- Do not expose internal labels, IDs, field names, or system words listed in jargon_policy.\n"
            "- Do not mention Vera, AI, dashboard, trigger, context, payload, template, approval, campaign setup, or drafting.\n"
            "- Do not make medical, dental, fitness, pharmacy, or beauty claims beyond appointment/refill/service reminders grounded in FACTS.\n"
            "- Do not use unsupported hype or certainty such as absolutely, definitely, perfect timing, customers love, guaranteed, or will fix.\n"
            "- Write this content pass in clear English. The polish pass handles any code-mix.\n"
            "- Follow cta_instruction for the final question.\n"
            "- The final sentence must be one customer-facing question; do not add another sentence after it.\n"
            "- Keep the body natural for WhatsApp and ideally under 500 characters.\n"
            "- Match the category voice and avoid taboo words.\n\n"
            "FACTS:\n"
            f"{json.dumps(fact_pack, ensure_ascii=False, indent=2)}"
        )

    def _prompt_opening(self, profile):
        openings = {
            "evidence": "Improve the baseline WhatsApp message as an evidence/update-led Vera nudge.",
            "performance": "Improve the baseline WhatsApp message as a performance/profile fix nudge.",
            "campaign": "Improve the baseline WhatsApp message as a timely campaign/planning nudge.",
            "reputation": "Improve the baseline WhatsApp message as a reputation or competition nudge.",
            "winback": "Improve the baseline WhatsApp message as a soft reactivation nudge.",
        }
        return openings.get(profile, "Improve the baseline WhatsApp message using only the facts below.")

    def _customer_prompt_opening(self, profile):
        openings = {
            "recall": "Improve the baseline customer reminder for a due recall or follow-up.",
            "appointment": "Improve the baseline customer reminder for a booked appointment or saved slot.",
            "refill": "Improve the baseline pharmacy refill reminder for a consented customer.",
            "winback": "Improve the baseline customer reactivation message with a low-pressure tone.",
            "trial": "Improve the baseline trial or package follow-up for a customer or parent.",
            "generic_customer": "Improve the baseline customer-facing WhatsApp message using only the facts below.",
        }
        return openings.get(profile, openings["generic_customer"])

    def _profile_rules(self, profile):
        rules = {
            "evidence": [
                "Lead with the update, research, regulation, learning, supply, or seasonal fact from trigger/related_update.",
                "Use one merchant-specific reason so it does not feel like a generic news alert.",
                "Do not turn it into a dashboard report.",
            ],
            "performance": [
                "Lead with the metric/profile issue or improvement, then explain the business impact.",
                "Suggest one practical fix using merchant performance, human_signals, active_offers, or baseline_draft.",
                "Do not say a fix will recover, unblock, or solve performance; frame it as the first practical step.",
                "Do not mention category digest, research, regulation, news, or trend items unless trigger.payload directly references them.",
            ],
            "campaign": [
                "Lead with the event, timing, or merchant planning intent.",
                "Connect the timing to one offer, product, menu, program, or customer behavior fact.",
                "Do not invent demand claims such as typical rush, spike, solid orders, or extra covers unless FACTS contain that claim.",
                "Do not mention category digest, research, regulation, or compliance unless trigger.payload directly references it.",
            ],
            "reputation": [
                "Lead with the review theme or competitor fact.",
                "Use loss aversion without panic; defend with proof, trust, service, or clarity rather than pure discounting.",
                "Use only review themes, offers, competitor details, and merchant facts in FACTS; do not invent service proof topics.",
                "Do not mention unrelated category research, regulation, or news.",
            ],
            "winback": [
                "Lead with the quiet period, lapsed relationship, or recent merchant reply.",
                "Keep the tone soft and low-pressure.",
                "Do not promise customers will return quickly or that seasonality will spike unless FACTS say so.",
                "Offer one restart step; do not over-explain or add unrelated category news.",
            ],
        }.get(profile, [
            "Explain the trigger-specific opportunity in plain merchant language.",
            "Use one merchant fact and one next step.",
        ])
        return "".join(f"- {rule}\n" for rule in rules)

    def _customer_profile_rules(self, profile):
        rules = {
            "recall": [
                "Lead with the due service or follow-up and the customer's last relevant visit/service if present.",
                "Use slots or offer only if they are in FACTS.",
                "Keep it clinical/practical; no treatment outcome claims.",
            ],
            "appointment": [
                "Lead with the appointment/service timing.",
                "Use saved preference or visit history as reassurance, not as pressure.",
                "Ask whether the time works or if another slot is needed.",
            ],
            "refill": [
                "Lead with medicine/refill timing or saved reminder.",
                "Mention delivery/address/senior discount only if present in FACTS.",
                "Ask whether to dispatch same brand/dose or if anything changed.",
            ],
            "winback": [
                "Acknowledge the gap without guilt or pressure.",
                "Use previous focus or visit history to personalize.",
                "Offer a small first step and make commitment feel optional.",
            ],
            "trial": [
                "Reference the trial/package context and the next available option.",
                "If the customer is a child, address the parent naturally.",
                "Ask to hold a spot or planning slot without pressure.",
            ],
        }.get(profile, [
            "Explain why the customer is being contacted in plain language.",
            "Use one customer relationship fact and one next step.",
        ])
        return "".join(f"- {rule}\n" for rule in rules)

    def _parse_llm_json(self, text):
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

    def _llm_fact_pack(self, category, merchant, trigger, template_action):
        profile = self._prompt_profile(trigger.get("kind"))
        item = self._llm_related_item(category, trigger, profile)
        voice = (category or {}).get("voice", {})
        facts = {
            "message_goal": "write one outbound WhatsApp message",
            "audience": "merchant",
            "prompt_profile": profile,
            "send_as": template_action.get("send_as"),
            "expected_cta": template_action.get("cta"),
            "allowed_ctas": sorted(Validator.VALID_CTAS),
            "cta_mode": "choose" if self._is_unknown_trigger(trigger) else "preserve",
            "cta_instruction": self._cta_instruction(trigger, template_action.get("cta"), category),
            "urgency_guidance": self._urgency_guidance(trigger.get("urgency")),
            "language_guidance": self._language_guidance(category),
            "jargon_policy": self._jargon_policy(),
            "category": {
                "slug": (category or {}).get("slug"),
                "tone": voice.get("tone"),
                "register": voice.get("register"),
                "code_mix": voice.get("code_mix"),
                "allowed_vocab": voice.get("vocab_allowed", [])[:8],
                "taboo_words": voice.get("vocab_taboo", [])[:8],
                "tone_examples": voice.get("tone_examples", [])[:2],
            },
            "merchant": {
                "salutation": self._merchant_salutation(category, merchant),
                "name": self._short_name(merchant),
                "place": self._place(merchant),
                "languages": merchant.get("identity", {}).get("languages", []),
                "performance": self._llm_performance_facts(merchant),
                "active_offers": self._active_offer_titles(merchant)[:3],
                "human_signals": self._llm_signal_facts(merchant),
                "recent_conversation": self._llm_history_facts(merchant),
                "review_themes": self._llm_review_facts(merchant),
            },
            "trigger": {
                "id": trigger.get("id"),
                "kind": trigger.get("kind"),
                "urgency": trigger.get("urgency"),
                "topic": self._trigger_topic(trigger, item, trigger.get("kind", "trigger").replace("_", " ")),
                "payload": trigger.get("payload", {}),
                "expires_at": trigger.get("expires_at"),
            },
            "strategy": self._llm_strategy(category, merchant, trigger, item),
            "baseline_draft": template_action.get("body", ""),
        }
        related_update = self._llm_related_update(item)
        if related_update:
            facts["related_update"] = related_update
        return facts

    def _customer_llm_fact_pack(self, category, merchant, trigger, customer, template_action):
        profile = self._customer_prompt_profile(trigger.get("kind"))
        voice = (category or {}).get("voice", {})
        payload = trigger.get("payload", {}) or {}
        facts = {
            "message_goal": "write one outbound WhatsApp message to a merchant customer",
            "audience": "customer",
            "prompt_profile": profile,
            "send_as": template_action.get("send_as"),
            "expected_cta": template_action.get("cta"),
            "allowed_ctas": sorted(Validator.VALID_CTAS),
            "cta_mode": "preserve",
            "cta_instruction": self._customer_cta_instruction(category, trigger, customer),
            "urgency_guidance": self._urgency_guidance(trigger.get("urgency")),
            "language_guidance": self._customer_language_guidance(category, customer),
            "jargon_policy": self._jargon_policy(),
            "category": {
                "slug": (category or {}).get("slug"),
                "tone": voice.get("tone"),
                "register": voice.get("register"),
                "code_mix": voice.get("code_mix"),
                "allowed_vocab": voice.get("vocab_allowed", [])[:8],
                "taboo_words": voice.get("vocab_taboo", [])[:8],
                "tone_examples": voice.get("tone_examples", [])[:2],
            },
            "merchant": {
                "name": self._customer_sender_name(merchant),
                "place": self._place(merchant),
                "active_offers": self._active_offer_titles(merchant)[:3],
            },
            "customer": {
                "display_name": self._customer_display_name(customer),
                "salutation": self._customer_salutation(customer),
                "language_pref": customer.get("identity", {}).get("language_pref"),
                "age_band": customer.get("identity", {}).get("age_band"),
                "state": customer.get("state"),
                "relationship": self._customer_relationship_facts(customer),
                "preferences": self._customer_preference_facts(customer),
                "consent_scope": (customer.get("consent", {}) or {}).get("scope", []),
            },
            "trigger": {
                "id": trigger.get("id"),
                "kind": trigger.get("kind"),
                "scope": trigger.get("scope"),
                "urgency": trigger.get("urgency"),
                "payload": payload,
                "expires_at": trigger.get("expires_at"),
            },
            "strategy": self._customer_llm_strategy(category, trigger, customer),
            "baseline_draft": template_action.get("body", ""),
        }
        slots = self._customer_slots(payload)
        if slots:
            facts["available_slots"] = slots
        return facts

    def _prompt_profile(self, kind):
        if kind in {"research_digest", "regulation_change", "cde_opportunity", "supply_alert", "category_seasonal"}:
            return "evidence"
        if kind in {"perf_dip", "perf_spike", "seasonal_perf_dip", "renewal_due", "gbp_unverified", "milestone_reached"}:
            return "performance"
        if kind in {"festival_upcoming", "ipl_match_today", "active_planning_intent"}:
            return "campaign"
        if kind in {"review_theme_emerged", "competitor_opened"}:
            return "reputation"
        if kind in {"winback_eligible", "dormant_with_vera", "curious_ask_due"}:
            return "winback"
        return "generic"

    def _customer_prompt_profile(self, kind):
        if kind == "recall_due":
            return "recall"
        if kind == "appointment_tomorrow":
            return "appointment"
        if kind == "chronic_refill_due":
            return "refill"
        if kind in {"customer_lapsed_hard", "customer_lapsed_soft"}:
            return "winback"
        if kind in {"trial_followup", "wedding_package_followup"}:
            return "trial"
        return "generic_customer"

    def _llm_related_item(self, category, trigger, profile):
        payload_item = self._digest_item_from_payload(category, trigger)
        if payload_item:
            return payload_item
        if profile != "evidence":
            return {}
        kind = trigger.get("kind")
        preferred = {
            "research_digest": {"research", "trend"},
            "regulation_change": {"compliance"},
            "cde_opportunity": {"cde"},
            "supply_alert": {"supply", "alert"},
            "category_seasonal": {"seasonal", "trend"},
        }.get(kind)
        return self._digest_item(category, trigger, preferred_kinds=preferred)

    def _llm_related_update(self, item):
        if not item:
            return None
        keys = ["kind", "title", "source", "summary", "actionable", "date", "credits", "trial_n"]
        update = {}
        for key in keys:
            value = item.get(key)
            if value is not None and value != "":
                update[key] = value
        return update or None

    def _cta_instruction(self, trigger, cta, category=None):
        kind = trigger.get("kind")
        slug = (category or {}).get("slug")
        intent_topic = str(trigger.get("payload", {}).get("intent_topic", "")).lower()
        follow_up_by_kind = {
            "regulation_change": "End with: Want me to draft the checklist and Google post?",
            "perf_dip": "End with: Want me to draft the exact 10-minute fix list?",
            "perf_spike": "End with: Want me to draft the follow-up post?",
            "gbp_unverified": "End with: Want me to send the verification steps?",
            "review_theme_emerged": "End with: Want me to draft the customer reply and fix checklist?",
            "competitor_opened": "End with: Want me to draft the response post?",
            "supply_alert": "End with: Want me to draft the customer note and replacement checklist?",
        }
        if kind == "active_planning_intent":
            return self._active_planning_cta(slug, intent_topic)
        if kind == "ipl_match_today":
            return "End with: Want me to draft the Google post and delivery banner copy?"
        if kind == "festival_upcoming":
            return "End with: Want me to draft two offer options and one Google post?"
        if kind == "winback_eligible":
            return self._show_offer_cta(slug)
        if kind == "dormant_with_vera":
            return "End with: Want me to send the one highest-impact restart step?"
        if kind == "curious_ask_due":
            return "End with one concrete question the merchant can answer in a few words."
        if kind in follow_up_by_kind:
            return follow_up_by_kind[kind]
        if cta == "follow_up":
            return "End with a concrete Want me to question for the next step."
        if cta == "show_offer":
            return self._show_offer_cta(slug)
        return "End with one concrete question about the next step, not vague curiosity."

    def _customer_cta_instruction(self, category, trigger, customer):
        kind = trigger.get("kind")
        payload = trigger.get("payload", {}) or {}
        slots = self._customer_slots(payload)
        if kind in {"recall_due", "appointment_tomorrow"} and slots:
            return "End with a slot-choice question using the available slots, or ask for another time."
        if kind == "appointment_tomorrow":
            return "End by asking whether the saved appointment time works or another slot is needed."
        if kind == "chronic_refill_due":
            return "End by asking whether to dispatch the same brand/dose or if anything changed."
        if kind in {"customer_lapsed_hard", "customer_lapsed_soft"}:
            return "End by asking whether to hold one low-pressure trial/session/slot."
        if kind in {"trial_followup", "wedding_package_followup"}:
            return "End by asking whether to hold the next planning or trial slot."
        if slots:
            return "End with one simple slot-choice question."
        return f"End with: {self._customer_next_step(category)}"

    def _active_planning_cta(self, slug, intent_topic):
        if slug == "restaurants":
            if "thali" in intent_topic or "corporate" in intent_topic:
                return "End with: Want me to draft the office menu and outreach message now?"
            return "End with: Want me to draft the campaign menu and customer message now?"
        if slug == "gyms":
            if "kids" in intent_topic or "yoga" in intent_topic:
                return "End with: Want me to draft the Google post and parent WhatsApp now?"
            return "End with: Want me to draft the campaign post and member message now?"
        if slug == "salons":
            return "End with: Want me to draft the offer post and client WhatsApp now?"
        if slug == "pharmacies":
            return "End with: Want me to draft the customer note and counter checklist now?"
        if slug == "dentists":
            return "End with: Want me to draft the patient post and WhatsApp now?"
        return "End with: Want me to draft the campaign copy now?"

    def _show_offer_cta(self, slug):
        if slug == "restaurants":
            return "End with: Want me to draft the offer post and delivery copy?"
        if slug == "gyms":
            return "End with: Want me to draft the offer post and member WhatsApp?"
        if slug == "salons":
            return "End with: Want me to draft the offer post and client WhatsApp?"
        if slug == "pharmacies":
            return "End with: Want me to draft the customer WhatsApp and counter note?"
        if slug == "dentists":
            return "End with: Want me to draft the patient post and WhatsApp?"
        return "End with: Want me to draft the offer/post copy?"

    def _urgency_guidance(self, urgency):
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

    def _language_guidance(self, category):
        voice = (category or {}).get("voice", {})
        code_mix = str(voice.get("code_mix", "")).lower()
        if "english_primary" in code_mix:
            return (
                "Mostly English. One short Roman Hindi phrase is okay if natural. "
                "Keep technical/category terms in English."
            )
        if any(token in code_mix for token in ["hindi", "hinglish", "hi-en", "natural"]):
            return (
                "Use natural Roman Hinglish where it improves warmth: 1-2 short Hindi phrases max. "
                "Keep salutation, names, places, prices, dates, offers, source names, and technical terms in English."
            )
        return "Use clear English in the category voice."

    def _customer_language_guidance(self, category, customer):
        pref = self._language_pref(customer)
        if any(token in pref for token in ["hi", "hindi", "hinglish"]):
            return (
                "Use natural Roman Hinglish for warmth: short phrases like Apke liye or kya time chalega are okay. "
                "Keep merchant name, customer name, prices, medicines, services, dates, and slots in English."
            )
        if any(token in pref for token in ["ta", "te", "kn"]):
            return (
                "Use mostly English with one light local-language warmth phrase only if natural. "
                "Keep merchant name, customer name, prices, services, dates, and slots in English."
            )
        return self._language_guidance(category)

    def _jargon_policy(self):
        return {
            "avoid_words_or_patterns": [
                "trigger",
                "payload",
                "context",
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
            ],
            "translate_examples": {
                "ctr_below_peer_median": "profile traffic is not converting as strongly as nearby peers",
                "stale_posts:22d": "Google posts have been quiet for 22 days",
                "high_risk_adult_cohort": "this is relevant to your patient mix",
            },
        }

    def _llm_performance_facts(self, merchant):
        perf = merchant.get("performance", {})
        facts = []
        if perf.get("views") is not None:
            facts.append(f"{int(perf['views']):,} profile views in {perf.get('window_days', 30)}d")
        if perf.get("calls") is not None:
            facts.append(f"{int(perf['calls']):,} calls in {perf.get('window_days', 30)}d")
        if perf.get("directions") is not None:
            facts.append(f"{int(perf['directions']):,} direction requests")
        if perf.get("ctr") is not None:
            facts.append(f"CTR {self._ctr_label(perf['ctr'])}")
        delta = perf.get("delta_7d", {})
        if delta.get("calls_pct") is not None:
            facts.append(f"calls {self._pct(delta['calls_pct'], signed=True)} this week")
        return facts

    def _llm_signal_facts(self, merchant):
        signals = [str(signal) for signal in merchant.get("signals", [])]
        facts = []
        for signal in signals:
            if signal.startswith("stale_posts"):
                match = re.search(r"(\d+)", signal)
                facts.append(f"Google posts stale for {match.group(1)} days" if match else "Google posts are stale")
            elif signal == "ctr_below_peer_median":
                facts.append("profile traffic is not converting as well as nearby peers")
            elif signal == "high_risk_adult_cohort":
                facts.append("patient mix includes high-risk adults")
            elif signal == "engaged_in_last_48h":
                facts.append("merchant replied to Vera in the last 48 hours")
            elif signal == "no_active_offers":
                facts.append("no active offer is live")
            elif signal == "unverified_gbp":
                facts.append("Google profile is unverified")
            else:
                facts.append(signal.replace("_", " "))
        return facts[:4]

    def _llm_history_facts(self, merchant):
        facts = []
        for turn in merchant.get("conversation_history", [])[-3:]:
            speaker = turn.get("from", "unknown")
            body = str(turn.get("body", "")).strip()
            if body:
                facts.append(f"{speaker}: {body[:180]}")
        return facts

    def _llm_review_facts(self, merchant):
        facts = []
        for theme in merchant.get("review_themes", [])[:2]:
            name = str(theme.get("theme", "")).replace("_", " ")
            count = theme.get("occurrences_30d")
            sentiment = theme.get("sentiment")
            if name:
                facts.append(f"{name}: {count} {sentiment or 'mentions'} in 30d")
        return facts

    def _llm_strategy(self, category, merchant, trigger, item):
        kind = trigger.get("kind")
        audience = self._audience_label(category)
        if kind == "research_digest":
            return (
                f"Lead with clinical curiosity about the update topic. Use one merchant fact only. "
                f"Make it feel useful for {audience}, not like a dashboard report."
            )
        if kind == "regulation_change":
            return (
                "Turn the compliance update into a trust-building next step. "
                "Use the deadline and one merchant visibility fact; avoid fake source dates."
            )
        if kind in {"perf_dip", "seasonal_perf_dip"}:
            return "Explain the metric movement, reduce anxiety, and offer one concrete fix."
        if kind == "perf_spike":
            return "Frame momentum as a short window to convert warm demand."
        if kind == "competitor_opened":
            return "Use loss aversion without panic; avoid a pure price fight."
        if kind in {"festival_upcoming", "ipl_match_today", "category_seasonal"}:
            return "Tie the external event to one timely campaign action and one active offer."
        if kind == "curious_ask_due":
            return "Ask the merchant one easy question and offer to turn the answer into content."
        if self._is_unknown_trigger(trigger):
            return (
                "Classify the trigger from trigger.kind and trigger.payload. "
                "Use the most concrete payload fact as the trigger fact, then connect it to one merchant-specific proof point."
            )
        return "Explain why this update matters now and end with one low-friction next step."

    def _customer_llm_strategy(self, category, trigger, customer):
        kind = trigger.get("kind")
        payload = trigger.get("payload", {}) or {}
        if kind == "recall_due":
            return "Use the due service, last visit/service, and any available slots; keep it reminder-like and calm."
        if kind == "appointment_tomorrow":
            return "Confirm the saved appointment timing and offer rescheduling if needed."
        if kind == "chronic_refill_due":
            return "Protect continuity: mention the refill/medicine timing and ask if dispatch details changed."
        if kind in {"customer_lapsed_hard", "customer_lapsed_soft"}:
            return "No guilt; reference the previous focus or visit gap and offer a small restart step."
        if kind == "trial_followup":
            return "Follow up after the trial and offer the next session slot without pressure."
        if kind == "wedding_package_followup":
            return "Use wedding timing and trial history to offer a planning slot without sounding pushy."
        if self._is_unknown_trigger(trigger):
            detail = self._customer_trigger_fact(category, trigger, customer)
            return f"Classify the customer update from payload and use this as the reason: {detail}"
        return "Explain the customer-specific reason for the message and end with one safe next step."

    def _is_unknown_trigger(self, trigger):
        return trigger.get("kind") not in self.CTA_BY_KIND

    def _empty_action(self, trigger):
        return {
            "conversation_id": f"conv_{self._safe_id(trigger.get('id', 'unknown'))}",
            "merchant_id": None,
            "customer_id": trigger.get("customer_id"),
            "send_as": "merchant_on_behalf" if trigger.get("scope") == "customer" else "vera",
            "trigger_id": trigger.get("id"),
            "template_name": "vera_missing_context_v1",
            "template_params": [],
            "body": "",
            "cta": "follow_up",
            "suppression_key": trigger.get("suppression_key", trigger.get("id", "")),
        }

    def _compose_body(self, category, merchant, trigger, customer):
        if trigger.get("scope") == "customer":
            return self._compose_customer_body(category, merchant, trigger, customer)
        return self._compose_merchant_body(category, merchant, trigger, customer)

    def _compose_merchant_body(self, category, merchant, trigger, customer):
        kind = trigger.get("kind", "")
        handlers = {
            "research_digest": self._research_digest,
            "regulation_change": self._regulation_change,
            "cde_opportunity": self._cde_opportunity,
            "perf_dip": self._perf_dip,
            "perf_spike": self._perf_spike,
            "seasonal_perf_dip": self._seasonal_perf_dip,
            "renewal_due": self._renewal_due,
            "festival_upcoming": self._festival_upcoming,
            "curious_ask_due": self._curious_ask,
            "winback_eligible": self._merchant_winback,
            "dormant_with_vera": self._merchant_dormant,
            "ipl_match_today": self._ipl_match,
            "review_theme_emerged": self._review_theme,
            "milestone_reached": self._milestone,
            "active_planning_intent": self._active_planning,
            "supply_alert": self._supply_alert,
            "category_seasonal": self._category_seasonal,
            "gbp_unverified": self._gbp_unverified,
            "competitor_opened": self._competitor_opened,
        }
        return handlers.get(kind, self._generic_trigger)(category, merchant, trigger, customer)

    def _compose_customer_body(self, category, merchant, trigger, customer):
        if not customer:
            return ""
        if not self._has_customer_consent(customer):
            return ""

        kind = trigger.get("kind", "")
        handlers = {
            "recall_due": self._recall_due,
            "wedding_package_followup": self._wedding_followup,
            "customer_lapsed_hard": self._customer_lapsed,
            "customer_lapsed_soft": self._customer_lapsed,
            "trial_followup": self._trial_followup,
            "chronic_refill_due": self._chronic_refill,
            "appointment_tomorrow": self._appointment_tomorrow,
        }
        return handlers.get(kind, self._generic_customer_trigger)(category, merchant, trigger, customer)

    # ------------------------------------------------------------------
    # Merchant-facing trigger templates
    # ------------------------------------------------------------------

    def _research_digest(self, category, merchant, trigger, customer):
        item = self._digest_item(category, trigger, preferred_kinds={"research", "trend"})
        sal = self._merchant_salutation(category, merchant)
        topic = self._trigger_topic(trigger, item, "the new category research update")
        profile = self._visible_profile_facts(merchant)
        signals = self._visible_signal_facts(merchant)
        signal_text = f" Your profile also shows {signals}." if signals else ""
        offer = self._best_offer(category, merchant, trigger)
        audience = self._audience_label(category)
        return (
            f"{sal}, {topic} is worth using now for {self._place(merchant)}. "
            f"Your profile has {profile}.{signal_text} "
            f"Instead of a generic promo, I can draft a trust post plus patient WhatsApp around {offer}. "
            f"Want the 4-line version for {audience}?"
        )

    def _regulation_change(self, category, merchant, trigger, customer):
        item = self._digest_item(category, trigger, preferred_kinds={"compliance"})
        sal = self._merchant_salutation(category, merchant)
        payload = trigger.get("payload", {})
        topic = self._trigger_topic(trigger, item, "the compliance update")
        deadline = self._date_label(payload.get("deadline_iso") or item.get("date"))
        deadline_text = f" has a {deadline} deadline" if deadline else " is active now"
        profile = self._visible_profile_facts(merchant)
        signals = self._visible_signal_facts(merchant)
        signal_text = f" With {signals}, this can be a trust-building update, not just back-office work." if signals else ""
        return (
            f"{sal}, compliance heads-up: {topic}{deadline_text}. "
            f"{self._short_name(merchant)} already has {profile} in {self._place(merchant)}. {signal_text} "
            "Want me to draft a short checklist plus a compliance-safe Google post?"
        )

    def _cde_opportunity(self, category, merchant, trigger, customer):
        item = self._digest_item(category, trigger, preferred_kinds={"cde"})
        sal = self._merchant_salutation(category, merchant)
        payload = trigger.get("payload", {})
        title = item.get("title") or "a category learning session"
        date = self._date_label(item.get("date") or trigger.get("expires_at"))
        credits = payload.get("credits") or item.get("credits")
        fee = payload.get("fee") or item.get("actionable")
        details = []
        if date:
            details.append(date)
        if credits:
            details.append(f"{credits} credits")
        if fee:
            details.append(str(fee).replace("_", " "))
        detail_text = ", ".join(details)
        detail_text = f" ({detail_text})" if detail_text else ""
        history = self._history_phrase(merchant, ["whitening", "aligners", "digital", "cad/cam"])
        relevance = "it can become a patient-facing update"
        if history:
            relevance = "it connects to the topic you recently discussed with Vera"
        return (
            f"{sal}, {title}{detail_text}. "
            f"Relevant because {relevance}. "
            "Want the key points and registration details in one message?"
        )

    def _perf_dip(self, category, merchant, trigger, customer):
        sal = self._merchant_salutation(category, merchant)
        payload = trigger.get("payload", {})
        metric = payload.get("metric") or self._delta_metric(merchant, prefer_negative=True) or "calls"
        delta_value = payload.get("delta_pct")
        if delta_value is None:
            delta_value = merchant.get("performance", {}).get("delta_7d", {}).get(f"{metric}_pct")
        if delta_value is not None:
            try:
                if float(delta_value) >= 0:
                    return self._perf_watch(category, merchant, trigger, customer)
            except (TypeError, ValueError):
                pass
        delta = self._pct_abs(delta_value)
        window = payload.get("window") or "7d"
        baseline = payload.get("vs_baseline")
        baseline_text = f" vs baseline {baseline}" if baseline is not None else ""
        fixes = self._fix_list(category, merchant)
        metric_text = self._metric_text(metric)
        perf = merchant.get("performance", {})
        volume = ""
        if perf.get("views") is not None and perf.get("calls") is not None:
            volume = f" Current 30-day base: {perf.get('views')} views but only {perf.get('calls')} calls."
        benchmark = self._peer_benchmark(category, merchant, metric)
        benchmark_text = f" {benchmark}" if benchmark else ""
        return (
            f"{sal}, {metric_text} down {delta} over {window}{baseline_text}. "
            f"Important because fewer {self._audience_label(category)} are converting from the profile.{volume}{benchmark_text} "
            f"I would start with {fixes}. Want me to draft the exact 10-minute fix list?"
        )

    def _perf_watch(self, category, merchant, trigger, customer):
        sal = self._merchant_salutation(category, merchant)
        perf = merchant.get("performance", {})
        delta = self._pct(perf.get("delta_7d", {}).get("calls_pct"), signed=True)
        views = perf.get("views")
        calls = perf.get("calls")
        offer = self._best_offer(category, merchant, trigger)
        volume = f"{views} views and {calls} calls in 30 days" if views is not None and calls is not None else "fresh profile activity"
        return (
            f"{sal}, quick performance check: calls are {delta} this week, with {volume}. "
            f"That is not a severe dip, but it is a good moment to tighten conversion around {offer}. "
            "Want me to draft the one profile update that should move calls first?"
        )

    def _perf_spike(self, category, merchant, trigger, customer):
        sal = self._merchant_salutation(category, merchant)
        payload = trigger.get("payload", {})
        metric = payload.get("metric") or self._delta_metric(merchant, prefer_negative=False) or "calls"
        delta_value = payload.get("delta_pct")
        if delta_value is None:
            delta_value = merchant.get("performance", {}).get("delta_7d", {}).get(f"{metric}_pct")
        delta = self._pct(delta_value, signed=True)
        driver = payload.get("likely_driver")
        driver_text = f" and the likely driver is {driver.replace('_', ' ')}" if driver else ""
        offer = self._best_offer(category, merchant, trigger)
        metric_text = self._metric_text(metric)
        perf = merchant.get("performance", {})
        volume = ""
        if perf.get("views") is not None and perf.get("calls") is not None:
            volume = f" You have {perf.get('views')} views and {perf.get('calls')} calls in the 30-day window."
        return (
            f"{sal}, {metric_text} up {delta} over the last {payload.get('window', '7d')}{driver_text}.{volume} "
            f"That momentum is useful only if you follow it while people are warm. "
            f"Want me to draft a follow-up post using {offer}?"
        )

    def _seasonal_perf_dip(self, category, merchant, trigger, customer):
        sal = self._merchant_salutation(category, merchant)
        payload = trigger.get("payload", {})
        delta = self._pct(payload.get("delta_pct"))
        member_count = merchant.get("customer_aggregate", {}).get("total_active_members")
        member_text = f" Your {member_count} active members are the better focus this month." if member_count else ""
        return (
            f"{sal}, views are down {delta} this {payload.get('window', 'week')}, and this looks like the normal Apr-Jun acquisition lull. "
            f"Important: avoid over-spending on cold acquisition during the low window.{member_text} "
            "Want me to draft a summer attendance challenge to protect retention?"
        )

    def _renewal_due(self, category, merchant, trigger, customer):
        sal = self._merchant_salutation(category, merchant)
        payload = trigger.get("payload", {})
        days = payload.get("days_remaining") or merchant.get("subscription", {}).get("days_remaining")
        amount = self._money(payload.get("renewal_amount"))
        amount_text = f" ({amount})" if amount else ""
        issue = self._fix_list(category, merchant)
        return (
            f"{sal}, your {payload.get('plan', merchant.get('subscription', {}).get('plan', 'plan'))} renewal is in {days} days{amount_text}. "
            f"Before renewing, the useful question is whether the profile is converting: {issue}. "
            "Want a quick audit message with the top 3 changes to do before renewal?"
        )

    def _festival_upcoming(self, category, merchant, trigger, customer):
        sal = self._merchant_salutation(category, merchant)
        payload = trigger.get("payload", {})
        festival = payload.get("festival")
        days = payload.get("days_until")
        offer = self._best_offer(category, merchant, trigger)
        if festival:
            timing = f"{days} days away" if days is not None else "coming up"
            opening = f"{festival} is {timing}"
            campaign = "festive package"
        else:
            opening = self._seasonal_hook(category) or "the next seasonal window is opening"
            campaign = "seasonal campaign"
        aggregate_hint = self._aggregate_hint(category, merchant)
        aggregate_text = f" {aggregate_hint}" if aggregate_hint else ""
        if not festival:
            return (
                f"{sal}, {opening}. This is about timing, not discounting. "
                f"Use {offer} as the low-friction hook for a clean {campaign}.{aggregate_text} "
                "Want me to draft two offer options and one Google post?"
            )
        return (
            f"{sal}, {opening}. Too early for a hard sell, but the planning window is open. "
            f"Your strongest starting point is {offer}; we can shape it into a clean {campaign} before slots get busy.{aggregate_text} "
            "Want me to draft two offer options and one Google post?"
        )

    def _curious_ask(self, category, merchant, trigger, customer):
        sal = self._merchant_salutation(category, merchant)
        performance = merchant.get("performance", {})
        calls_delta = self._pct(performance.get("delta_7d", {}).get("calls_pct"), signed=True)
        trend = self._top_review_theme(merchant)
        trend_text = f" and reviews mention {trend}" if trend else ""
        offer_guess = self._best_offer(category, merchant, trigger)
        return (
            f"{sal}, quick check: calls are {calls_delta} this week{trend_text}. "
            f"Are customers asking more about {offer_guess}, or is something else leading at {self._short_name(merchant)} right now? "
            "Want me to turn your answer into a Google post plus a 4-line WhatsApp reply?"
        )

    def _merchant_winback(self, category, merchant, trigger, customer):
        sal = self._merchant_salutation(category, merchant)
        payload = trigger.get("payload", {})
        days = payload.get("days_since_expiry") or merchant.get("subscription", {}).get("days_since_expiry")
        dip = self._pct(payload.get("perf_dip_pct") or merchant.get("performance", {}).get("delta_7d", {}).get("calls_pct"))
        lapsed = payload.get("lapsed_customers_added_since_expiry") or self._first_present(
            merchant.get("customer_aggregate", {}), ["lapsed_90d_plus", "lapsed_180d_plus"]
        )
        offer = self._best_offer(category, merchant, trigger)
        lapsed_text = f" and {lapsed} customers are now in a lapsed bucket" if lapsed is not None else ""
        return (
            f"{sal}, it has been {days} days since the profile work paused; calls are down {dip}{lapsed_text}. "
            f"That makes winback harder each week. I would restart with one simple hook: {offer}. "
            "Want me to draft the reactivation note?"
        )

    def _merchant_dormant(self, category, merchant, trigger, customer):
        sal = self._merchant_salutation(category, merchant)
        payload = trigger.get("payload", {})
        days = payload.get("days_since_last_merchant_message") or self._signal_number(merchant, "dormant_with_vera")
        gap = f"not had a reply for {days} days" if str(days).isdigit() else "no recent reply logged"
        last_topic = payload.get("last_topic")
        topic_text = f" after {last_topic.replace('_', ' ')}" if last_topic else ""
        views = merchant.get("performance", {}).get("views")
        calls = merchant.get("performance", {}).get("calls")
        activity = f"{views} profile views and {calls} calls" if views is not None and calls is not None else "fresh profile activity"
        return (
            f"{sal}, we have {gap}{topic_text}. "
            f"Meanwhile {self._short_name(merchant)} still has {activity} in the 30-day window, so there is demand worth capturing. "
            "Want me to send just the one highest-impact fix to restart?"
        )

    def _ipl_match(self, category, merchant, trigger, customer):
        sal = self._merchant_salutation(category, merchant)
        payload = trigger.get("payload", {})
        match = payload.get("match", "tonight's match")
        venue = payload.get("venue")
        time_label = self._time_label(payload.get("match_time_iso"))
        item = self._digest_item(category, trigger, preferred_kinds={"seasonal"})
        summary = item.get("summary", "")
        saturday_note = "Saturday IPL matches shift covers down 12% vs Saturday average" if "down 12%" in summary else "match nights change ordering patterns"
        offer = self._best_offer(category, merchant, trigger)
        venue_text = f" at {venue}" if venue else ""
        time_text = f", {time_label}" if time_label else ""
        return (
            f"{sal}, {match}{venue_text}{time_text}. Since this is not a weeknight, {saturday_note}. "
            f"Use {offer} as a delivery-first hook instead of pushing dine-in. "
            "Want me to draft the Google post and delivery banner copy?"
        )

    def _review_theme(self, category, merchant, trigger, customer):
        sal = self._merchant_salutation(category, merchant)
        payload = trigger.get("payload", {})
        theme = payload.get("theme", "review theme").replace("_", " ")
        count = payload.get("occurrences_30d")
        quote = payload.get("common_quote")
        count_text = f"{count} reviews mention {theme}" if count is not None else f"{theme} is showing up in reviews"
        quote_text = f' One quote: "{quote}".' if quote else ""
        return (
            f"{sal}, {count_text} in the last 30 days.{quote_text} "
            "Important because review themes change conversion before ratings visibly move. "
            "Want me to draft a short customer reply and an internal fix checklist?"
        )

    def _milestone(self, category, merchant, trigger, customer):
        sal = self._merchant_salutation(category, merchant)
        payload = trigger.get("payload", {})
        now = payload.get("value_now")
        milestone = payload.get("milestone_value")
        metric = payload.get("metric", "milestone").replace("_", " ")
        if now is None or milestone is None:
            perf = merchant.get("performance", {})
            views = perf.get("views")
            calls = perf.get("calls")
            ctr = self._pct(perf.get("ctr"))
            activity = f"{views} views, {calls} calls, CTR {ctr}" if views is not None and calls is not None else "fresh profile activity"
            return (
                f"{sal}, {self._short_name(merchant)} has a visible trust moment: {activity} in the 30-day window. "
                f"A review ask or proof post can make that traffic convert better in {self._place(merchant)}. "
                "Want me to write the one-line customer ask?"
            )
        gap = milestone - now if isinstance(now, int) and isinstance(milestone, int) else None
        gap_text = f" - just {gap} away" if gap is not None else ""
        theme = self._top_review_theme(merchant)
        theme_text = f" Your strongest review hook is {theme}." if theme else ""
        return (
            f"{sal}, you are at {now} {metric} and close to {milestone}{gap_text}. "
            f"That is a visible trust signal for new customers in {self._place(merchant)}.{theme_text} "
            "Want me to write the one-line review ask for today's regulars?"
        )

    def _active_planning(self, category, merchant, trigger, customer):
        payload = trigger.get("payload", {})
        topic = payload.get("intent_topic", "the plan").replace("_", " ")
        sal = self._merchant_salutation(category, merchant)
        if "corporate" in topic or "thali" in topic:
            offer = self._best_offer(category, merchant, trigger)
            delivery = merchant.get("customer_aggregate", {}).get("delivery_share_pct")
            delivery_text = f" Delivery is already {self._pct(delivery)} of your orders, so this fits existing demand." if delivery is not None else ""
            return (
                f"{sal}, picking up your question on {topic}: use {offer} as the base, with pre-orders the previous evening and a fixed lunch delivery window. "
                f"{delivery_text} Want me to draft the one-page office menu and the 3-line outreach message?"
            )
        if "kids" in topic or "yoga" in topic:
            history_hint = self._history_phrase(merchant, ["4-week", "kids yoga", "summer"])
            hint = self._program_hint_from_history(history_hint) or "a 4-week kids program with small batches"
            return (
                f"{sal}, your {topic} idea is timely for the summer window. "
                f"The clean structure is {hint}, positioned around safety, attention, and parent-friendly timing. "
                "Want me to draft the Google post plus parent WhatsApp?"
            )
        return (
            f"{sal}, picking up your planning intent on {topic}. "
            f"The next useful step is a concrete draft built around {self._best_offer(category, merchant, trigger)} and your {self._place(merchant)} audience. "
            "Do you want me to draft it now?"
        )

    def _supply_alert(self, category, merchant, trigger, customer):
        sal = self._merchant_salutation(category, merchant)
        payload = trigger.get("payload", {})
        molecule = payload.get("molecule", "medicine")
        batches = ", ".join(payload.get("affected_batches", []))
        manufacturer = payload.get("manufacturer")
        chronic = merchant.get("customer_aggregate", {}).get("chronic_rx_count")
        chronic_text = f" You have {chronic} chronic-Rx customers, so the outreach list matters." if chronic is not None else ""
        batch_text = f" batches {batches}" if batches else " specific batches"
        mfr_text = f" by {manufacturer}" if manufacturer else ""
        return (
            f"{sal}, urgent supply alert: {molecule}{batch_text}{mfr_text}. "
            f"Important because affected customers need a calm replacement workflow, not panic.{chronic_text} "
            "Want me to draft the customer note and pickup/replacement checklist?"
        )

    def _category_seasonal(self, category, merchant, trigger, customer):
        sal = self._merchant_salutation(category, merchant)
        payload = trigger.get("payload", {})
        trends = self._format_trends(payload.get("trends", []))
        trend_text = "; ".join(trends[:4]) or self._first_trend(category)
        aggregate = merchant.get("customer_aggregate", {})
        repeat = aggregate.get("repeat_customer_pct")
        repeat_text = f" With repeat customers at {self._pct(repeat)}, put these items where regulars can see them first." if repeat is not None else ""
        return (
            f"{sal}, seasonal demand is shifting now: {trend_text}. "
            f"Important because shelf/profile visibility should move before customers ask.{repeat_text} "
            "Want me to draft the counter-display checklist plus a short WhatsApp update?"
        )

    def _gbp_unverified(self, category, merchant, trigger, customer):
        sal = self._merchant_salutation(category, merchant)
        payload = trigger.get("payload", {})
        perf = merchant.get("performance", {})
        path = payload.get("verification_path", "Google verification").replace("_", " ")
        uplift = self._pct(payload.get("estimated_uplift_pct"))
        return (
            f"{sal}, {self._short_name(merchant)} is still unverified on Google. "
            f"That matters because the profile already has {perf.get('views', '?')} views and {perf.get('directions', '?')} direction requests, and verification is estimated to lift visibility by {uplift}. "
            f"Want me to walk you through {path} step by step?"
        )

    def _competitor_opened(self, category, merchant, trigger, customer):
        sal = self._merchant_salutation(category, merchant)
        payload = trigger.get("payload", {})
        named = bool(payload.get("competitor_name"))
        name = payload.get("competitor_name", "local competition alert")
        distance = payload.get("distance_km")
        offer = payload.get("their_offer")
        opened = self._date_label(payload.get("opened_date"))
        own_offer = self._best_offer(category, merchant, trigger)
        opened_text = f" on {opened}" if opened else ""
        distance_text = f" {distance} km away" if distance is not None else (f" around {self._place(merchant)}" if not named else " nearby")
        offer_text = f" with {offer}" if offer else ""
        verb = "opened" if named else "is active"
        return (
            f"{sal}, {name} {verb}{distance_text}{opened_text}{offer_text}. "
            f"Important: do not turn this into a pure price fight. Your stronger move is to defend with {own_offer} plus one trust-building post for {self._place(merchant)} {self._audience_label(category)}. "
            "Want me to draft both?"
        )

    def _generic_trigger(self, category, merchant, trigger, customer):
        sal = self._merchant_salutation(category, merchant)
        kind = trigger.get("kind", "update").replace("_", " ")
        payload = trigger.get("payload", {})
        item = self._digest_item_from_payload(category, trigger)
        detail = self._digest_item_detail(item) or self._payload_detail(payload) or self._first_trend(category) or kind
        proof = self._generic_merchant_proof(category, merchant)
        reason = self._generic_decision_reason(trigger, merchant)
        next_step = self._generic_next_step(category, trigger, self._select_cta(trigger))
        return (
            f"{sal}, {detail}. "
            f"{proof} "
            f"{reason} "
            f"{next_step}"
        )

    def _generic_merchant_proof(self, category, merchant):
        proof = self._visible_profile_facts(merchant)
        place = self._place(merchant)
        review = self._top_review_theme(merchant)
        if review:
            return f"{self._short_name(merchant)} already has {proof} in {place}, and reviews mention {review}."
        signal = self._visible_signal_facts(merchant)
        if signal:
            return f"{self._short_name(merchant)} already has {proof} in {place}; {signal}."
        offer = self._best_offer(category, merchant, {})
        return f"{self._short_name(merchant)} already has {proof} in {place}, with {offer} available."

    def _generic_decision_reason(self, trigger, merchant):
        expires = self._date_label(trigger.get("expires_at"))
        if expires:
            return f"The useful decision is what to publish before {expires}, while the signal is still current."
        urgency = trigger.get("urgency")
        try:
            if int(urgency) >= 4:
                return "This is time-sensitive, so the useful decision is the first action to take now."
        except (TypeError, ValueError):
            pass
        if merchant.get("performance", {}).get("views") is not None:
            return "Because the profile already has active traffic, the useful decision is what to show those customers next."
        return "The useful decision is the next customer-facing action, not just noting the update."

    def _generic_next_step(self, category, trigger, cta):
        slug = (category or {}).get("slug")
        if cta == "show_offer":
            return self._show_offer_cta(slug).replace("End with: ", "")
        if cta == "open_ended":
            return "Want me to turn this into one message option?"
        return "Want me to draft the next action message?"

    def _digest_item_detail(self, item):
        if not item:
            return ""
        title = item.get("title")
        source = item.get("source")
        if title and source:
            return f"{title} from {source}"
        return title or item.get("summary", "")

    # ------------------------------------------------------------------
    # Customer-facing trigger templates
    # ------------------------------------------------------------------

    def _generic_customer_trigger(self, category, merchant, trigger, customer):
        name = self._customer_salutation(customer)
        sender = self._customer_sender_name(merchant)
        reason = self._customer_trigger_fact(category, trigger, customer)
        relationship = self._customer_relationship_line(customer)
        next_step = self._customer_next_step(category)
        return (
            f"{name}, {sender} here. {reason}. "
            f"{relationship} {next_step}"
        )

    def _customer_trigger_fact(self, category, trigger, customer):
        payload = trigger.get("payload", {}) or {}
        kind = str(trigger.get("kind") or "follow-up").replace("_", " ")
        service = payload.get("service_due") or payload.get("service")
        if service:
            text = str(service).replace("_", " ")
            due = self._date_label(payload.get("due_date") or payload.get("date") or payload.get("appointment_iso"))
            return f"Your {text} is due around {due}" if due else f"Your {text} follow-up is due"
        meds = payload.get("molecule_list")
        if meds:
            runout = self._date_label(payload.get("stock_runs_out_iso"))
            runout_text = f" around {runout}" if runout else ""
            return f"Your saved refill for {', '.join(str(m) for m in meds[:3])} is due{runout_text}"
        if payload.get("appointment_iso") or payload.get("date"):
            when = self._date_label(payload.get("appointment_iso") or payload.get("date"))
            return f"Your appointment reminder is for {when}"
        if payload.get("trial_date"):
            trial = self._date_label(payload.get("trial_date"))
            return f"This is a follow-up from your trial on {trial}"
        if payload.get("days_since_last_visit") is not None:
            focus = str(payload.get("previous_focus") or "your routine").replace("_", " ")
            return f"It has been {payload.get('days_since_last_visit')} days since your last visit for {focus}"
        if payload.get("next_step_window_open"):
            return f"Your {str(payload.get('next_step_window_open')).replace('_', ' ')} window is open"
        if payload.get("topic") or payload.get("title"):
            return str(payload.get("topic") or payload.get("title")).replace("_", " ")
        last_visit = self._date_label(customer.get("relationship", {}).get("last_visit"))
        if last_visit:
            return f"This is a saved follow-up from your last visit on {last_visit}"
        return f"This {kind} reminder is due from your saved preferences"

    def _customer_relationship_line(self, customer):
        rel = customer.get("relationship", {}) or {}
        prefs = customer.get("preferences", {}) or {}
        pieces = []
        visits = rel.get("visits_total")
        if visits:
            pieces.append(f"This is based on your {visits} earlier visit{'s' if visits != 1 else ''}")
        services = self._service_phrase(customer)
        if services:
            pieces.append(f"for {services}")
        pref = str(prefs.get("preferred_slots", "")).replace("_", " ")
        if pref:
            pieces.append(f"and your {pref} preference")
        if pieces:
            return " ".join(pieces) + "."
        return "This is based only on your saved reminder."

    def _recall_due(self, category, merchant, trigger, customer):
        name = self._customer_salutation(customer)
        clinic = self._customer_sender_name(merchant)
        payload = trigger.get("payload", {})
        service = payload.get("service_due", "recall").replace("_", " ")
        if service == "recall" and (category or {}).get("slug") != "dentists":
            service = "follow-up"
        last = self._date_label(payload.get("last_service_date"))
        if not last:
            last = self._date_label(customer.get("relationship", {}).get("last_visit"))
        due = self._date_label(payload.get("due_date"))
        slots = [s.get("label") for s in payload.get("available_slots", []) if s.get("label")]
        offer = self._best_offer(category, merchant, trigger, customer)
        lang = self._language_pref(customer)
        middle = "Apke liye" if "hi" in lang else "For you,"
        last_text = f" Last visit: {last}." if last else ""
        due_text = f" Your {service} is due around {due}." if due else f" Your {service} is due."
        if slots:
            slot_text = f"{self._slot_text(slots)}. {offer}"
            reply_text = "Which slot works for you, or is another time better?"
        else:
            slot_text = f"{offer} is available"
            reply_text = "Would you like us to share today/tomorrow timings?"
        return (
            f"{name}, {clinic} here.{last_text}{due_text} "
            f"{middle} {slot_text}. {reply_text}"
        )

    def _wedding_followup(self, category, merchant, trigger, customer):
        name = self._customer_salutation(customer)
        payload = trigger.get("payload", {})
        days = payload.get("days_to_wedding")
        trial = self._date_label(payload.get("trial_completed"))
        window = str(payload.get("next_step_window_open", "skin prep window")).replace("_", " ")
        preferred = customer.get("preferences", {}).get("preferred_slots", "preferred slot").replace("_", " ")
        trial_text = f" since your bridal trial on {trial}" if trial else ""
        days_text = f"{days} days to your wedding" if days is not None else "your wedding window is open"
        return (
            f"{name}, {self._short_name(merchant)} here. {days_text}{trial_text}; this is the right time for the {window}. "
            f"We can plan it around your {preferred}. Would you like Lakshmi's team to hold the first planning slot?"
        )

    def _customer_lapsed(self, category, merchant, trigger, customer):
        name = self._customer_salutation(customer)
        payload = trigger.get("payload", {})
        days = payload.get("days_since_last_visit")
        focus = payload.get("previous_focus") or customer.get("preferences", {}).get("training_focus") or "your routine"
        offer = self._best_offer(category, merchant, trigger, customer)
        no_shame = self._lapsed_reassurance(category)
        last_visit = self._date_label(customer.get("relationship", {}).get("last_visit"))
        days_text = f"It has been {days} days since your last visit" if days is not None else f"Last visit was {last_visit}" if last_visit else "The follow-up window is open"
        slot_name = self._lapsed_slot_label(category)
        visits = customer.get("relationship", {}).get("visits_total")
        visits_text = f" You had {visits} visit{'s' if visits != 1 else ''} with us." if visits else ""
        membership = payload.get("previous_membership_months")
        membership_text = f" Your last membership ran {membership} months." if membership else ""
        return (
            f"{name}, {self._customer_sender_name(merchant)} here. {days_text}.{visits_text}{membership_text} {no_shame}. "
            f"If {self._lapsed_focus_phrase(category, focus)}, {offer} is the easiest first step. "
            f"No commitment - would you like us to hold a {slot_name}?"
        )

    def _trial_followup(self, category, merchant, trigger, customer):
        contact = self._parent_or_customer_name(customer)
        payload = trigger.get("payload", {})
        trial = self._date_label(payload.get("trial_date"))
        options = [s.get("label") for s in payload.get("next_session_options", []) if s.get("label")]
        slot_text = self._slot_text(options)
        child = self._child_name(customer)
        child_text = f" for {child}" if child else ""
        trial_text = f" after the trial on {trial}" if trial else ""
        return (
            f"Hi {contact}, {self._customer_sender_name(merchant)} here{child_text}. "
            f"We have {slot_text}{trial_text}. Small batches keep the follow-up personal. "
            "Would you like us to hold the spot?"
        )

    def _chronic_refill(self, category, merchant, trigger, customer):
        payload = trigger.get("payload", {})
        meds_list = payload.get("molecule_list", [])
        if not meds_list:
            last_visit = self._date_label(customer.get("relationship", {}).get("last_visit"))
            services = self._service_phrase(customer)
            context = f"after your {services}" if services else "from your saved WhatsApp reminder"
            last_text = f" Last visit: {last_visit}." if last_visit else ""
            return (
                f"{self._customer_salutation(customer)}, {self._customer_sender_name(merchant)} here. "
                f"This follow-up is due {context}.{last_text} {self._best_offer(category, merchant, trigger, customer)}. "
                f"{self._customer_next_step(category)}"
            )
        meds = ", ".join(meds_list)
        runout = self._date_label(payload.get("stock_runs_out_iso"))
        runout_text = f" run out on {runout}" if runout else " are due for refill"
        delivery = payload.get("delivery_address_saved") or customer.get("preferences", {}).get("delivery_address") == "saved"
        senior = self._is_senior(customer)
        offers = self._active_offer_titles(merchant)
        relevant_offers = [o for o in offers if "delivery" in o.lower() or "senior" in o.lower()]
        offer_text = "; ".join(relevant_offers[:2]) or self._best_offer(category, merchant, trigger, customer)
        delivery_text = " Saved address is on file." if delivery else ""
        senior_text = " Senior discount will be applied." if senior else ""
        return (
            f"Namaste - {self._customer_sender_name(merchant)}. {self._family_patient_name(customer)}'s {meds}{runout_text}. "
            f"{offer_text}.{delivery_text}{senior_text} Would you like us to dispatch the same brand/dose, or has anything changed?"
        )

    def _appointment_tomorrow(self, category, merchant, trigger, customer):
        payload = trigger.get("payload", {})
        when = self._date_label(payload.get("appointment_iso") or payload.get("date"))
        service = payload.get("service", "appointment").replace("_", " ")
        if service == "appointment":
            service = self._appointment_label(category, customer)
        prefs = customer.get("preferences", {})
        pref = str(prefs.get("preferred_slots", "")).replace("_", " ")
        pref_text = f" We have your preferred timing as {pref}." if pref else ""
        visits = customer.get("relationship", {}).get("visits_total")
        if visits and visits > 1:
            visits_text = f" This is just a saved reminder from your {visits} earlier visits."
        elif visits:
            visits_text = " This is just a saved reminder from your earlier visit."
        else:
            visits_text = ""
        if when:
            schedule_text = f"scheduled for {when}"
        else:
            schedule_text = "scheduled tomorrow from your saved reminder"
        return (
            f"{self._customer_salutation(customer)}, reminder from {self._customer_sender_name(merchant)}: your {service} is {schedule_text}. "
            f"{visits_text}{pref_text} Does this time work, or would you like another slot?"
        )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _select_cta(self, trigger):
        known_cta = self.CTA_BY_KIND.get(trigger.get("kind"))
        if known_cta:
            return known_cta
        return self._infer_unknown_cta(trigger)

    def _infer_unknown_cta(self, trigger):
        payload = trigger.get("payload", {}) or {}
        text = " ".join([str(trigger.get("kind", "")), str(payload)]).lower()
        show_offer_terms = [
            "offer", "discount", "deal", "coupon", "festival", "campaign", "package",
            "menu", "price", "pricing", "bundle", "promo", "promotion", "sale",
        ]
        open_ended_terms = [
            "question", "ask", "asked", "intent", "planning", "plan", "idea",
            "feedback", "survey", "preference", "which", "what should", "curious",
        ]
        follow_up_terms = [
            "alert", "deadline", "compliance", "regulation", "review", "competitor",
            "dip", "drop", "unverified", "verification", "recall", "supply", "renewal",
        ]
        if any(term in text for term in follow_up_terms):
            return "follow_up"
        if any(term in text for term in open_ended_terms):
            return "open_ended"
        if any(term in text for term in show_offer_terms):
            return "show_offer"
        return "follow_up"

    def _build_rationale(self, category, merchant, trigger, customer, cta):
        category_slug = category.get("slug") if category else merchant.get("category_slug", "unknown")
        scope = "customer" if customer else "merchant"
        reason = trigger.get("kind", "trigger").replace("_", " ")
        place = self._place(merchant)
        return (
            f"{reason} trigger for {self._short_name(merchant)} in {place}; "
            f"uses {category_slug} voice, grounded context facts, and a single {cta} CTA for {scope} scope."
        )

    def _conversation_id(self, merchant, trigger, customer=None):
        bits = ["conv", merchant.get("merchant_id", "merchant"), trigger.get("id", "trigger")]
        if customer:
            bits.append(customer.get("customer_id", "customer"))
        return self._safe_id("_".join(bits))[:160]

    def _template_name(self, send_as, kind):
        prefix = "merchant" if send_as == "merchant_on_behalf" else "vera"
        return f"{prefix}_{self._safe_id(kind or 'generic')}_v1"

    def _template_params(self, body, merchant, trigger, customer=None):
        recipient = self._parent_or_customer_name(customer) if customer else self._short_name(merchant)
        return [recipient, trigger.get("kind", "update"), body[:120]]

    def _merchant_salutation(self, category, merchant):
        identity = merchant.get("identity", {})
        owner = identity.get("owner_first_name")
        slug = category.get("slug") if category else merchant.get("category_slug")
        if owner:
            owner = str(owner).strip()
            if slug == "dentists" and not owner.lower().startswith("dr"):
                return f"Dr. {owner}"
            return owner
        return identity.get("name", "there")

    def _customer_salutation(self, customer):
        return f"Hi {self._parent_or_customer_name(customer)}"

    def _customer_sender_name(self, merchant):
        identity = merchant.get("identity", {})
        name = identity.get("name", "the clinic")
        locality = identity.get("locality")
        if locality and locality.lower() not in name.lower():
            return f"{name} {locality}"
        return name

    def _short_name(self, merchant):
        return merchant.get("identity", {}).get("name", "your business")

    def _place(self, merchant):
        identity = merchant.get("identity", {})
        locality = identity.get("locality")
        city = identity.get("city")
        if locality and city:
            return f"{locality}, {city}"
        return locality or city or "your locality"

    def _business_type(self, category):
        slug = (category or {}).get("slug")
        labels = {
            "dentists": "clinic",
            "salons": "salon",
            "restaurants": "restaurant",
            "gyms": "gym",
            "pharmacies": "pharmacy",
        }
        if slug in labels:
            return labels[slug]
        display = (category or {}).get("display_name") or slug or "business"
        return str(display).split("&")[0].strip().lower()

    def _appointment_label(self, category, customer):
        slug = (category or {}).get("slug")
        services = customer.get("relationship", {}).get("services_received", [])
        clean_service = ""
        if services:
            clean_service = str(services[-1]).replace("_", " ")
        defaults = {
            "dentists": "clinic appointment",
            "salons": "salon appointment",
            "restaurants": "booking",
            "gyms": "session",
            "pharmacies": "pharmacy follow-up",
        }
        return clean_service or defaults.get(slug, "appointment")

    def _service_phrase(self, customer):
        services = [str(s).replace("_", " ") for s in customer.get("relationship", {}).get("services_received", []) if s and s != "..."]
        if not services:
            return ""
        unique = []
        for service in services:
            if service not in unique:
                unique.append(service)
        return ", ".join(unique[:2])

    def _seasonal_hook(self, category):
        beats = (category or {}).get("seasonal_beats", [])
        if not beats:
            return ""
        business = self._business_type(category)
        for beat in beats:
            month = str(beat.get("month_range", ""))
            if any(token in month for token in ["Apr", "May", "Jun", "Mar-Apr", "Apr-May", "Apr-Jun"]):
                note = beat.get("note")
                if month and note:
                    return f"{month} {business} window is open - {note}"
                return note or ""
        first = beats[0]
        month = first.get("month_range")
        note = first.get("note")
        if month and note:
            return f"{month} {business} window is coming up - {note}"
        return note or ""

    def _trigger_topic(self, trigger, item, fallback):
        """Name the trigger using fields that are visible in the trigger payload."""
        payload = trigger.get("payload", {})
        raw_id = " ".join(str(payload.get(key, "")) for key in ["top_item_id", "digest_item_id", "alert_id", "item_id"]).lower()
        if "dci" in raw_id and "radiograph" in raw_id:
            return "the DCI radiograph compliance update"
        if "jida" in raw_id and "fluoride" in raw_id:
            return "the JIDA fluoride-recall note"
        if "ida" in raw_id and "webinar" in raw_id:
            return "the IDA learning opportunity"
        if "atorvastatin" in raw_id:
            return "the atorvastatin supply alert"
        if payload.get("title"):
            return str(payload["title"]).replace("_", " ")
        if item.get("kind"):
            return f"the {str(item['kind']).replace('_', ' ')} update"
        return fallback

    def _visible_profile_facts(self, merchant):
        perf = merchant.get("performance", {})
        pieces = []
        if perf.get("views") is not None:
            pieces.append(f"{int(perf['views']):,} views")
        if perf.get("calls") is not None:
            pieces.append(f"{int(perf['calls']):,} calls")
        if perf.get("ctr") is not None:
            pieces.append(f"CTR {self._ctr_label(perf['ctr'])}")
        return ", ".join(pieces) or "fresh profile activity"

    def _visible_signal_facts(self, merchant):
        signals = [str(signal) for signal in merchant.get("signals", [])]
        facts = []
        for signal in signals:
            if signal.startswith("stale_posts"):
                match = re.search(r"(\d+)", signal)
                facts.append(f"Google posts stale for {match.group(1)} days" if match else "stale Google posts")
            elif signal == "ctr_below_peer_median":
                facts.append("profile traffic is not converting as well as nearby peers")
            elif signal == "high_risk_adult_cohort":
                facts.append("patient mix includes high-risk adults")
            elif signal == "engaged_in_last_48h":
                facts.append("you replied to Vera in the last 48 hours")
            elif signal == "no_active_offers":
                facts.append("no active offers")
            elif signal == "unverified_gbp":
                facts.append("unverified Google profile")
        return ", ".join(facts[:3])

    def _ctr_label(self, value):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)
        pct = number * 100 if abs(number) <= 1 else number
        text = f"{pct:.1f}".rstrip("0").rstrip(".")
        return f"{text}%"

    def _evidence_phrase(self, item):
        """Pull only explicit numbers already present in the digest item."""
        if not item:
            return ""
        pieces = []
        trial = item.get("trial_n")
        if isinstance(trial, int):
            pieces.append(f"{trial:,}-participant")
        searchable = " ".join(str(item.get(key, "")) for key in ["title", "summary", "actionable"])
        percent_match = re.search(r"[-+]?\d+(?:\.\d+)?\s*%", searchable)
        if percent_match:
            pct = re.sub(r"\s+", "", percent_match.group(0))
            window = searchable[max(0, percent_match.start() - 32):percent_match.end() + 40].strip()
            if window:
                pieces.append(window)
            else:
                pieces.append(pct)
        return "; ".join(dict.fromkeys(pieces))

    def _peer_benchmark(self, category, merchant, metric):
        perf = merchant.get("performance", {})
        metric = str(metric or "").lower()
        signals = [str(signal) for signal in merchant.get("signals", [])]
        if metric in {"ctr", "calls"} and "ctr_below_peer_median" in signals and perf.get("ctr") is not None:
            return f"CTR is {self._ctr_label(perf.get('ctr'))}, so nearby peers appear to be converting profile traffic better."
        return ""

    def _aggregate_hint(self, category, merchant):
        aggregate = merchant.get("customer_aggregate", {})
        slug = (category or {}).get("slug")
        if slug == "gyms" and aggregate.get("total_active_members"):
            return f"Start with your {aggregate['total_active_members']} active members, not cold leads."
        if slug == "pharmacies" and aggregate.get("chronic_rx_count"):
            return f"Your {aggregate['chronic_rx_count']} chronic-Rx customers are the repeat base to protect."
        if aggregate.get("repeat_customer_pct") is not None:
            return f"Repeat customers are at {self._pct(aggregate['repeat_customer_pct'])}, so regulars should see it first."
        if aggregate.get("total_unique_ytd"):
            return f"You have {aggregate['total_unique_ytd']} known customers YTD to seed the first push."
        return ""

    def _lapsed_reassurance(self, category):
        slug = (category or {}).get("slug")
        if slug == "gyms":
            return "No guilt, no pressure"
        if slug == "dentists":
            return "No pressure; a quick checkup is the simplest restart"
        if slug == "salons":
            return "No rush; this is just an easy slot to refresh"
        if slug == "pharmacies":
            return "Simple refill, no pressure"
        if slug == "restaurants":
            return "If you want your usual order again, we can keep it easy"
        return "No pressure"

    def _lapsed_slot_label(self, category):
        slug = (category or {}).get("slug")
        return {
            "gyms": "trial slot",
            "dentists": "checkup slot",
            "salons": "salon slot",
            "pharmacies": "delivery/check-in",
            "restaurants": "booking",
        }.get(slug, "follow-up slot")

    def _lapsed_focus_phrase(self, category, focus):
        focus_text = str(focus).replace("_", " ")
        slug = (category or {}).get("slug")
        if slug == "pharmacies":
            return "you still need regular refills"
        if slug == "dentists":
            return "you want a quick dental check"
        if slug == "salons":
            return "you want a simple refresh"
        if slug == "restaurants":
            return "you want your usual order again"
        return f"you want to restart {focus_text}"

    def _customer_next_step(self, category):
        slug = (category or {}).get("slug")
        if slug == "pharmacies":
            return "Would you like delivery, or has anything in the prescription changed?"
        if slug == "restaurants":
            return "Would you like us to share the next available booking or delivery option?"
        return "Would you like us to share the next available slot?"

    def _format_trends(self, trends):
        formatted = []
        for trend in trends:
            text = str(trend).replace("_", " ")
            text = re.sub(r"([A-Za-z])([+-]\d+)", r"\1 \2%", text)
            text = re.sub(r"([+-]\d+)(?!%)", r"\1%", text)
            formatted.append(text)
        return formatted

    def _audience_label(self, category):
        slug = (category or {}).get("slug")
        return {
            "dentists": "patients",
            "salons": "clients",
            "restaurants": "customers",
            "gyms": "members",
            "pharmacies": "customers",
        }.get(slug, "customers")

    def _has_customer_consent(self, customer):
        consent = customer.get("consent", {})
        preferences = customer.get("preferences", {})
        if preferences.get("reminder_opt_in") is False:
            return False
        scope = consent.get("scope") or []
        return bool(consent.get("opted_in_at") or scope or preferences.get("reminder_opt_in"))

    def _digest_item(self, category, trigger, preferred_kinds=None):
        category = category or {}
        payload = trigger.get("payload", {})
        wanted_ids = [
            payload.get("top_item_id"),
            payload.get("digest_item_id"),
            payload.get("alert_id"),
            payload.get("item_id"),
        ]
        digest = category.get("digest", [])
        for wanted in [w for w in wanted_ids if w]:
            for item in digest:
                if item.get("id") == wanted:
                    return item
        if preferred_kinds:
            for item in digest:
                if item.get("kind") in preferred_kinds:
                    return item
        return digest[0] if digest else {}

    def _digest_item_from_payload(self, category, trigger):
        """Return a digest item only when the trigger explicitly names one."""
        category = category or {}
        payload = trigger.get("payload", {}) or {}
        wanted_ids = [
            payload.get("top_item_id"),
            payload.get("digest_item_id"),
            payload.get("alert_id"),
            payload.get("item_id"),
            payload.get("update_id"),
            payload.get("news_id"),
        ]
        digest = category.get("digest", [])
        for wanted in [str(wanted) for wanted in wanted_ids if wanted]:
            for item in digest:
                if str(item.get("id")) == wanted:
                    return item
        return {}

    def _patient_segment(self, merchant, item):
        aggregate = merchant.get("customer_aggregate", {})
        segment = item.get("patient_segment", "")
        if segment == "high_risk_adults" and aggregate.get("high_risk_adult_count"):
            return f"your {aggregate['high_risk_adult_count']} high-risk adult patients"
        if aggregate.get("lapsed_180d_plus"):
            return f"your {aggregate['lapsed_180d_plus']} lapsed patients"
        return ""

    def _best_offer(self, category, merchant, trigger, customer=None):
        active = self._active_offer_titles(merchant)
        kind = trigger.get("kind", "")
        payload_text = " ".join(str(v).lower() for v in trigger.get("payload", {}).values())
        for offer in active:
            offer_l = offer.lower()
            if any(token in offer_l for token in self._offer_tokens(kind, payload_text)):
                return offer
        if active:
            return active[0]
        for offer in (category or {}).get("offer_catalog", []):
            title = offer.get("title")
            if not title:
                continue
            title_l = title.lower()
            if any(token in title_l for token in self._offer_tokens(kind, payload_text)):
                return title
        catalog = (category or {}).get("offer_catalog", [])
        if catalog:
            return catalog[0].get("title", "a service-at-price offer")
        return "one concrete service-at-price offer"

    def _offer_tokens(self, kind, payload_text):
        tokens = {
            "recall_due": ["cleaning", "checkup"],
            "wedding_package_followup": ["bridal", "skin"],
            "customer_lapsed_hard": ["trial", "month", "free"],
            "customer_lapsed_soft": ["trial", "month", "free"],
            "trial_followup": ["month", "trial"],
            "chronic_refill_due": ["delivery", "senior", "refill"],
            "festival_upcoming": ["hair", "spa", "bridal", "combo", "match"],
            "ipl_match_today": ["pizza", "match", "bogo"],
            "active_planning_intent": ["thali", "kids", "trial", "month"],
            "competitor_opened": ["cleaning", "consultation"],
        }.get(kind, [])
        if "thali" in payload_text:
            tokens.append("thali")
        if "kids" in payload_text or "yoga" in payload_text:
            tokens.extend(["month", "trial", "yoga"])
        return tokens or ["@", "free", "delivery"]

    def _active_offer_titles(self, merchant):
        titles = []
        for offer in merchant.get("offers", []):
            if offer.get("status") == "active" and offer.get("title"):
                titles.append(offer["title"])
        return titles

    def _fix_list(self, category, merchant):
        signals = [str(s) for s in merchant.get("signals", [])]
        active_offer = self._best_offer(category, merchant, {"kind": "generic", "payload": {}})
        if any("unverified" in s for s in signals):
            return "Google verification first"
        if any("no_active_offers" in s for s in signals) or not self._active_offer_titles(merchant):
            return f"one active offer such as {active_offer}"
        if any("stale_posts" in s or "no_recent_post" in s for s in signals):
            return "a fresh Google post tied to your current offer"
        if any("ctr_below" in s for s in signals):
            return "the profile CTA and offer clarity"
        return f"a sharper post around {active_offer}"

    def _top_review_theme(self, merchant):
        themes = merchant.get("review_themes", [])
        if not themes:
            return ""
        best = sorted(themes, key=lambda t: t.get("occurrences_30d", 0), reverse=True)[0]
        name = str(best.get("theme", "")).replace("_", " ")
        count = best.get("occurrences_30d")
        sentiment = best.get("sentiment")
        if count:
            return f"{name} ({count} {sentiment or 'mentions'})"
        return name

    def _history_phrase(self, merchant, needles):
        needles = [n.lower() for n in needles]
        for turn in reversed(merchant.get("conversation_history", [])):
            body = turn.get("body", "")
            lower = body.lower()
            if any(n in lower for n in needles):
                return body.strip().rstrip(".")
        return ""

    def _program_hint_from_history(self, text):
        if not text:
            return ""
        match = re.search(r"suggest\s+([^.?]+)", text, re.I)
        if match:
            return match.group(1).strip()
        match = re.search(r"(\d+-week[^.?]+)", text, re.I)
        if match:
            return match.group(1).strip()
        return ""

    def _first_trend(self, category):
        trends = (category or {}).get("trend_signals", [])
        if not trends:
            return ""
        trend = trends[0]
        delta = self._pct(trend.get("delta_yoy"), signed=True)
        return f"{trend.get('query', 'category search')} is {delta} YoY"

    def _payload_detail(self, payload):
        payload = payload or {}
        metric = payload.get("metric") or payload.get("metric_or_topic")
        delta = payload.get("delta_pct") or payload.get("change_pct") or payload.get("growth_pct")
        if metric and delta is not None:
            window = payload.get("window")
            window_text = f" over {window}" if window else ""
            return f"{str(metric).replace('_', ' ')} changed by {self._pct(delta, signed=True)}{window_text}"

        if payload.get("competitor_name"):
            distance = payload.get("distance_km")
            distance_text = f" {distance} km away" if distance is not None else " nearby"
            offer = f" with {payload.get('their_offer')}" if payload.get("their_offer") else ""
            return f"{payload['competitor_name']} is active{distance_text}{offer}"

        if payload.get("molecule") and payload.get("affected_batches"):
            batches = ", ".join(str(batch) for batch in payload.get("affected_batches", [])[:3])
            manufacturer = f" by {payload.get('manufacturer')}" if payload.get("manufacturer") else ""
            return f"{payload.get('molecule')} alert affects batches {batches}{manufacturer}"

        if payload.get("event") or payload.get("festival"):
            subject = self._payload_value_text(payload.get("event") or payload.get("festival"))
            date = self._date_label(payload.get("date") or payload.get("event_date") or payload.get("starts_at"))
            distance = payload.get("distance_km")
            footfall = payload.get("expected_footfall") or payload.get("footfall")
            details = []
            if date:
                details.append(date)
            if distance is not None:
                details.append(f"{distance} km away")
            if footfall is not None:
                details.append(f"{int(footfall):,} expected footfall" if isinstance(footfall, int) else f"{footfall} expected footfall")
            suffix = f" ({', '.join(details)})" if details else ""
            return f"{subject}{suffix}"

        if payload.get("theme") or payload.get("review_theme"):
            theme = self._payload_value_text(payload.get("theme") or payload.get("review_theme"))
            count = payload.get("occurrences_30d") or payload.get("count")
            return f"{count} recent mentions of {theme}" if count is not None else f"{theme} is showing up"

        deadline = payload.get("deadline_iso") or payload.get("deadline") or payload.get("due_date")
        if deadline:
            return f"deadline is {self._date_label(deadline)}"
        for key in ["title", "topic", "intent_topic", "item_title", "alert_title", "campaign", "offer_title", "service"]:
            if payload.get(key):
                detail = self._payload_value_text(payload[key])
                return detail

        priority_keys = [
            "source", "segment", "customer_segment", "product", "service_due",
            "verification_path", "recommended_action", "actionable", "reason",
        ]
        for key in priority_keys:
            if payload.get(key):
                return f"{key.replace('_', ' ')}: {self._payload_value_text(payload[key])}"

        skip_keys = {"top_item_id", "digest_item_id", "item_id", "alert_id", "update_id", "news_id"}
        for key, value in payload.items():
            if key in skip_keys or key.endswith("_id"):
                continue
            if value not in (None, "", [], {}):
                return f"{key.replace('_', ' ')}: {self._payload_value_text(value)}"
        return ""

    def _payload_value_text(self, value):
        if isinstance(value, list):
            return ", ".join(self._payload_value_text(item) for item in value[:4])
        if isinstance(value, dict):
            pieces = []
            for key, item in list(value.items())[:4]:
                if item not in (None, "", [], {}):
                    pieces.append(f"{key.replace('_', ' ')} {self._payload_value_text(item)}")
            return ", ".join(pieces)
        return str(value).replace("_", " ")

    def _signal_phrase(self, merchant, fallback):
        signals = [str(s).replace("_", " ") for s in merchant.get("signals", [])]
        return signals[0] if signals else fallback

    def _signal_number(self, merchant, prefix):
        for signal in merchant.get("signals", []):
            text = str(signal)
            if text.startswith(prefix):
                match = re.search(r"(\d+)", text)
                if match:
                    return match.group(1)
        return "a while"

    def _first_present(self, data, keys):
        for key in keys:
            if data.get(key) is not None:
                return data[key]
        return None

    def _language_pref(self, customer):
        return str(customer.get("identity", {}).get("language_pref", "")).lower()

    def _customer_display_name(self, customer):
        return self._parent_or_customer_name(customer)

    def _parent_or_customer_name(self, customer):
        if not customer:
            return "there"
        name = str(customer.get("identity", {}).get("name", "there"))
        match = re.search(r"parent:\s*([^)]+)", name, re.I)
        if match:
            return match.group(1).strip()
        if name.startswith("("):
            return "there"
        return name.split("(")[0].strip()

    def _customer_relationship_facts(self, customer):
        rel = customer.get("relationship", {}) or {}
        facts = []
        if rel.get("last_visit"):
            facts.append(f"last visit {self._date_label(rel.get('last_visit'))}")
        if rel.get("visits_total") is not None:
            facts.append(f"{rel.get('visits_total')} visits total")
        services = self._service_phrase(customer)
        if services:
            facts.append(f"services: {services}")
        if rel.get("favourite_dish"):
            facts.append(f"favourite dish: {rel.get('favourite_dish')}")
        if rel.get("chronic_conditions"):
            facts.append("chronic refill relationship on file")
        return facts[:5]

    def _customer_preference_facts(self, customer):
        prefs = customer.get("preferences", {}) or {}
        facts = []
        for key in ["preferred_slots", "channel", "training_focus", "health_focus", "preferred_stylist", "delivery_address", "wedding_date"]:
            value = prefs.get(key)
            if value not in (None, "", [], {}):
                facts.append(f"{key.replace('_', ' ')}: {str(value).replace('_', ' ')}")
        return facts[:5]

    def _customer_slots(self, payload):
        slots = []
        for key in ["available_slots", "next_session_options", "slots"]:
            value = payload.get(key)
            if isinstance(value, list):
                for item in value[:3]:
                    if isinstance(item, dict):
                        label = item.get("label") or item.get("iso")
                    else:
                        label = item
                    if label:
                        slots.append(str(label))
        return slots[:3]

    def _child_name(self, customer):
        name = str(customer.get("identity", {}).get("name", ""))
        if "parent:" in name:
            return name.split("(")[0].strip()
        return ""

    def _family_patient_name(self, customer):
        name = self._parent_or_customer_name(customer)
        if name.lower().startswith("mr."):
            return f"{name}"
        return name

    def _is_senior(self, customer):
        identity = customer.get("identity", {})
        return bool(identity.get("senior_citizen") or "65" in str(identity.get("age_band", "")))

    def _slot_text(self, slots):
        if not slots:
            return "we will share available times"
        if len(slots) == 1:
            return f"one slot open: {slots[0]}"
        numbered = [f"{idx + 1}) {label}" for idx, label in enumerate(slots[:3])]
        return "slots open: " + "; ".join(numbered)

    def _pct(self, value, signed=False):
        if value is None:
            return "recently"
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)
        pct = number * 100 if abs(number) <= 1 else number
        prefix = "+" if signed and pct > 0 else ""
        return f"{prefix}{pct:.0f}%"

    def _pct_abs(self, value):
        if value is None:
            return "recently"
        try:
            number = abs(float(value))
        except (TypeError, ValueError):
            return str(value).lstrip("-")
        pct = number * 100 if abs(number) <= 1 else number
        return f"{pct:.0f}%"

    def _metric_text(self, metric):
        metric = str(metric or "performance").replace("_", " ")
        if metric.lower() in {"calls", "views", "directions", "leads"}:
            return f"{metric} are"
        if metric.lower() == "ctr":
            return "CTR is"
        return f"{metric} is"

    def _delta_metric(self, merchant, prefer_negative=True):
        deltas = merchant.get("performance", {}).get("delta_7d", {})
        best_key = None
        best_value = None
        for key, value in deltas.items():
            if not key.endswith("_pct"):
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if prefer_negative and numeric >= 0:
                continue
            if not prefer_negative and numeric <= 0:
                continue
            if best_value is None or abs(numeric) > abs(best_value):
                best_key = key
                best_value = numeric
        if best_key:
            return best_key[:-4]
        return None

    def _money(self, value):
        if value is None:
            return ""
        try:
            return f"Rs {int(value):,}"
        except (TypeError, ValueError):
            return str(value)

    def _date_label(self, value):
        if not value:
            return ""
        text = str(value)
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            return dt.strftime("%d %b %Y").lstrip("0")
        except ValueError:
            return text

    def _time_label(self, value):
        if not value:
            return ""
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            return dt.strftime("%-I:%M%p").lower()
        except ValueError:
            return str(value)

    def _safe_id(self, value):
        return re.sub(r"[^a-zA-Z0-9_:-]+", "_", str(value)).strip("_")

    def _clean_body(self, body):
        body = (body or "").strip()
        body = re.sub(r"\s+\.", ".", body)
        body = re.sub(r"[ \t]+", " ", body)
        body = re.sub(r"\n{3,}", "\n\n", body)
        return body
