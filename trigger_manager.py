from datetime import datetime, timezone


class TriggerManager:
    def __init__(self, store):
        self.store = store

    def choose_trigger(self, available_trigger_ids):
        ranked = self.rank_triggers(available_trigger_ids)
        return ranked[0] if ranked else None

    def rank_triggers(self, available_trigger_ids, limit=20):
        """Return active trigger ids ordered by urgency.

        The judge already passes triggers it considers available for this tick.
        We therefore trust that hint instead of dropping seed triggers whose
        expires_at is in the past relative to a developer's local clock.
        """
        triggers = []
        for trigger_id in available_trigger_ids:
            payload = self.store.get_context("trigger", trigger_id)
            if not payload:
                continue
            urgency = payload.get("urgency", 0)
            triggers.append((urgency, trigger_id))

        if not triggers:
            return []

        triggers.sort(key=lambda item: (-item[0], item[1]))
        return [trigger_id for _, trigger_id in triggers[:limit]]

    def _is_expired(self, trigger_payload):
        expires_at = trigger_payload.get("expires_at")
        if not expires_at:
            return False
        try:
            iso = expires_at.replace("Z", "+00:00")
            expires = datetime.fromisoformat(iso)
            now = datetime.now(timezone.utc)
            if expires.tzinfo is None:
                expires = expires.replace(tzinfo=timezone.utc)
            return expires < now
        except Exception:
            return False
