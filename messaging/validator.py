import re


class Validator:
    REQUIRED_ACTION_FIELDS = {
        "conversation_id",
        "merchant_id",
        "send_as",
        "trigger_id",
        "template_name",
        "template_params",
        "body",
        "cta",
        "suppression_key",
    }
    VALID_SEND_AS = {"vera", "merchant_on_behalf"}
    VALID_OUTBOUND_CTAS = {"open_ended", "show_offer", "follow_up"}
    VALID_REPLY_CTAS = {
        "open_ended",
        "binary_yes_no",
        "binary_confirm_cancel",
        "multi_choice_slot",
        "none",
    }
    VALID_CTAS = VALID_OUTBOUND_CTAS
    VALID_REPLY_ACTIONS = {"send", "wait", "end"}
    URL_RE = re.compile(r"\b(?:https?://|www\.)\S+", re.I)
    MONEY_RE = re.compile(r"(?:₹\s*[\d,]+|rs\.?\s*[\d,]+)", re.I)
    BOILERPLATE_RE = re.compile(
        r"(?:^|\n)\s*(?:here(?:'| i)s the message|here is the whatsapp|"
        r"whatsapp message:|```|json|claude error|sample vera message)",
        re.I,
    )
    REPLY_COMMAND_RE = re.compile(r"\breply\s+(?:yes|confirm|with)\b|\b(?:answer|send)\s+(?:yes|confirm)\b", re.I)
    MERCHANT_CTA_RE = re.compile(
        r"\b(?:want me to|do you want me to|would you like|want the|want a|should i|can i|which|does this|are customers)\b",
        re.I,
    )
    LOW_SIGNAL_PHRASES = [
        "a while",
        "one suitable slot",
        "metric_or_topic",
        "category window",
        "no panic signal",
        "customers are still seeing",
        "trigger",
        "payload",
        "suppression_key",
        "send_as",
        "template_name",
        "template_params",
        "ctr_below_peer_median",
        "high_risk_adult_cohort",
        "stale_posts:",
        "delta_7d",
        "digest item",
        "signals show",
        "trg_",
        "d_2026",
        "top_item_id",
    ]

    def validate(self, action, category=None, merchant=None, trigger=None, customer=None):
        problems = []
        if not isinstance(action, dict):
            return False, ["action must be a dictionary"]

        missing = sorted(self.REQUIRED_ACTION_FIELDS - set(action.keys()))
        if missing:
            problems.append("missing required fields: " + ", ".join(missing))

        body = action.get("body", "") or ""
        body_stripped = body.strip()
        if not body_stripped:
            problems.append("message body is empty")
        elif len(body_stripped) < 40:
            problems.append("message body is too short to explain the trigger")

        send_as = action.get("send_as")
        if send_as not in self.VALID_SEND_AS:
            problems.append("send_as must be 'vera' or 'merchant_on_behalf'")
        if trigger:
            scope = trigger.get("scope")
            if scope == "customer" and send_as != "merchant_on_behalf":
                problems.append("customer-scope trigger must send_as merchant_on_behalf")
            if scope == "merchant" and send_as != "vera":
                problems.append("merchant-scope trigger must send_as vera")

        cta = action.get("cta")
        if cta not in self.VALID_OUTBOUND_CTAS:
            problems.append("cta must be one of open_ended, show_offer, follow_up")

        if action.get("template_params") is not None and not isinstance(action.get("template_params"), list):
            problems.append("template_params must be a list")

        if self.URL_RE.search(body):
            problems.append("urls are not allowed in outbound body")

        if self.BOILERPLATE_RE.search(body):
            problems.append("body contains assistant boilerplate or formatting")
        if self.REPLY_COMMAND_RE.search(body):
            problems.append("body contains reply-command CTA wording")
        lower_body = body.lower()
        for phrase in self.LOW_SIGNAL_PHRASES:
            if phrase in lower_body:
                problems.append(f"low-signal phrase detected: {phrase}")

        if category:
            self._validate_category_voice(body, category, problems)

        if merchant or trigger or customer or category:
            self._validate_grounding(body, category, merchant, trigger, customer, problems)

        if trigger and trigger.get("scope") == "merchant":
            self._validate_merchant_cta(body_stripped, problems)

        if problems:
            return False, problems
        return True, []

    def validate_reply(self, response, state=None):
        problems = []
        if not isinstance(response, dict):
            return False, ["reply response must be a dictionary"]

        action = response.get("action")
        if action not in self.VALID_REPLY_ACTIONS:
            problems.append("reply action must be one of send, wait, end")

        if action == "send":
            body = (response.get("body") or "").strip()
            if not body:
                problems.append("reply send body is empty")
            elif len(body) < 25:
                problems.append("reply send body is too short")
            if response.get("cta") not in self.VALID_REPLY_CTAS:
                problems.append("reply send cta must be one of binary_yes_no, binary_confirm_cancel, multi_choice_slot, open_ended, none")
            if self.URL_RE.search(body):
                problems.append("reply body contains a URL")
            if self.BOILERPLATE_RE.search(body):
                problems.append("reply body contains assistant boilerplate")
            if self.REPLY_COMMAND_RE.search(body):
                problems.append("reply body contains reply-command CTA wording")
            lower_body = body.lower()
            for phrase in self.LOW_SIGNAL_PHRASES:
                if phrase in lower_body:
                    problems.append(f"reply low-signal phrase detected: {phrase}")
            previous = [str(text).strip().lower() for text in (state or {}).get("sent_bodies", [])]
            if body.lower() in previous:
                problems.append("reply body repeats a previous body in this conversation")

        if action == "wait":
            wait_seconds = response.get("wait_seconds")
            if not isinstance(wait_seconds, int) or wait_seconds <= 0:
                problems.append("wait action must include positive integer wait_seconds")

        if not response.get("rationale"):
            problems.append("reply response must include rationale")

        return (not problems), problems

    def _validate_merchant_cta(self, body, problems):
        tail = body[-220:].strip()
        if not tail.endswith("?"):
            problems.append("merchant body should end with a concrete question CTA")
        elif not self.MERCHANT_CTA_RE.search(tail):
            problems.append("merchant CTA should use a direct next-step question")

    def _validate_category_voice(self, body, category, problems):
        lower_body = body.lower()
        taboo_words = category.get("voice", {}).get("vocab_taboo", [])
        for taboo in taboo_words:
            taboo = str(taboo).strip()
            if not taboo:
                continue
            normalized = taboo.lower()
            if "(" in normalized:
                normalized = normalized.split("(", 1)[0].strip()
            if normalized and normalized in lower_body:
                problems.append(f"taboo word detected: {taboo}")

        generic_banned = [
            "guaranteed",
            "miracle",
            "best in city",
            "100% safe",
            "completely cure",
            "viral guarantee",
            "pakka result",
            "100% fayda",
            "sure-shot",
            "sure shot",
            "pakka fayda",
            "guaranteed result",
        ]
        for phrase in generic_banned:
            if phrase in lower_body:
                problems.append(f"unsupported claim detected: {phrase}")

    def _validate_grounding(self, body, category, merchant, trigger, customer, problems):
        context_text = self._context_text(category, merchant, trigger, customer)
        body_lower = body.lower()

        if "[claude error:" in body_lower or "traceback" in body_lower:
            problems.append("body contains an implementation error")

        for money in self.MONEY_RE.findall(body):
            normalized = self._normalize_money(money)
            if normalized and normalized not in context_text:
                problems.append(f"money amount not grounded in context: {money}")

        known_source_terms = {
            "jida",
            "dci",
            "ida",
            "icmr",
            "cdsco",
            "dcgi",
            "gst council",
            "fssai",
            "zomato",
            "swiggy",
            "practo",
            "google trends",
        }
        for term in known_source_terms:
            if term in body_lower and term not in context_text:
                problems.append(f"source mention not present in context: {term}")

        if "competitor" in body_lower or "opened" in body_lower:
            kind = (trigger or {}).get("kind")
            if kind != "competitor_opened" and "competitor" in body_lower:
                problems.append("competitor claim used outside competitor trigger")

        if customer:
            customer_issue = self._customer_send_as_issue(body, merchant)
            if customer_issue:
                problems.append(customer_issue)

    def _customer_send_as_issue(self, body, merchant):
        merchant_name = (merchant or {}).get("identity", {}).get("name")
        if not merchant_name:
            return ""
        lower_body = body.lower()
        if merchant_name.lower().split()[0] not in lower_body:
            return "customer-facing body should identify the merchant"
        if "vera" in lower_body:
            return "customer-facing body should not mention Vera"
        if "want me to draft" in lower_body or "approval draft" in lower_body:
            return "customer-facing body should not expose merchant-assistant workflow"
        return ""

    def _context_text(self, category, merchant, trigger, customer):
        parts = []
        for obj in [category, merchant, trigger, customer]:
            if obj:
                parts.append(str(obj).lower())
        text = " ".join(parts)
        text = text.replace("₹", "rs ")
        text = re.sub(r"rs\s*([\d,]+)", lambda m: "rs " + m.group(1).replace(",", ""), text)
        return text

    def _normalize_money(self, value):
        normalized = value.lower().replace("₹", "rs ")
        normalized = normalized.replace(".", "")
        normalized = re.sub(r"\s+", " ", normalized)
        normalized = re.sub(r"rs\s*([\d,]+)", lambda m: "rs " + m.group(1).replace(",", ""), normalized)
        return normalized.strip()
