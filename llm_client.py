import os
import time

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None


class LLMClient:
    def __init__(self, model_name=None):
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.model_name = model_name or os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
        self.client = None
        if Anthropic and self.api_key:
            try:
                timeout = float(os.environ.get("ANTHROPIC_TIMEOUT", "5.5"))
            except ValueError:
                timeout = 5.5
            try:
                self.client = Anthropic(api_key=self.api_key, timeout=timeout)
            except TypeError:
                self.client = Anthropic(api_key=self.api_key)

    def available(self):
        return self.client is not None

    def generate(self, prompt, system=None, max_tokens=450, temperature=0.2):
        if self.client:
            return self._call_claude(prompt, system, max_tokens, temperature)
        return self._fallback_response(prompt)

    def _call_claude(self, prompt, system=None, max_tokens=450, temperature=0.2):
        started = time.perf_counter()
        try:
            kwargs = {
                "model": self.model_name,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            }
            if system:
                kwargs["system"] = system
            response = self.client.messages.create(**kwargs)
            self._debug(f"Anthropic call model={self.model_name} max_tokens={max_tokens} took {self._elapsed_ms(started)}ms")
            return response.content[0].text.strip()
        except Exception as exc:
            self._debug(f"Anthropic call model={self.model_name} failed after {self._elapsed_ms(started)}ms: {exc}")
            return f"[Claude error: {str(exc)}]"

    def _debug(self, message):
        if os.environ.get("VERA_DEBUG_LLM", "1").lower() not in {"0", "false", "no"}:
            formatted = f"[LLM] {message}"
            print(formatted, flush=True)
            self._append_debug_log(formatted)

    def _elapsed_ms(self, started):
        return int((time.perf_counter() - started) * 1000)

    def _append_debug_log(self, message):
        log_file = os.environ.get("VERA_LLM_LOG_FILE", "/tmp/vera_llm_debug.log")
        try:
            with open(log_file, "a", encoding="utf-8") as handle:
                handle.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {message}\n")
        except OSError:
            pass

    def _fallback_response(self, prompt):
        lines = [line for line in prompt.splitlines() if line.strip()]
        if not lines:
            return "Hello, this is a sample Vera message."

        lead = lines[0][:120]
        return f"Hi there! Here’s an AI-generated message based on the current merchant and trigger: {lead}"
