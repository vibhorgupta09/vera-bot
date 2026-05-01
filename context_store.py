import threading

VALID_SCOPES = {"category", "merchant", "customer", "trigger"}


class ContextStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._contexts = {scope: {} for scope in VALID_SCOPES}

    def add_context(self, scope, context_id, version, payload):
        if scope not in VALID_SCOPES:
            return False, {"reason": "invalid_scope", "details": f"scope must be one of {sorted(VALID_SCOPES)}."}
        with self._lock:
            current = self._contexts[scope].get(context_id)
            if current and version < current["version"]:
                return False, {"reason": "stale_version", "current_version": current["version"]}
            if current and version == current["version"]:
                return True, {"idempotent": True, "current_version": current["version"]}
            self._contexts[scope][context_id] = {"version": version, "payload": payload}
        return True, {}

    def get_context(self, scope, context_id):
        value = self._contexts.get(scope, {}).get(context_id)
        return value["payload"] if value else None

    def counts(self):
        return {scope: len(items) for scope, items in self._contexts.items()}

    def list_context_ids(self, scope):
        return list(self._contexts.get(scope, {}).keys())

    def clear(self):
        with self._lock:
            self._contexts = {scope: {} for scope in VALID_SCOPES}
