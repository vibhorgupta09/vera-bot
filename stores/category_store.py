import json
import os
from glob import glob


class CategoryStore:
    def __init__(self, categories_path):
        self.categories_path = categories_path
        self._categories = {}
        self._load_categories()

    def _load_categories(self):
        if not os.path.isdir(self.categories_path):
            return
        for path in glob(os.path.join(self.categories_path, "*.json")):
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                    slug = payload.get("slug")
                    if slug:
                        self._categories[slug] = payload
            except Exception:
                continue

    def get_category(self, slug):
        if not slug:
            return None
        return self._categories.get(slug)

    def list_categories(self):
        return list(self._categories.keys())
