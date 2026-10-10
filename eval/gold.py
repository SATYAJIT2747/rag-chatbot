import os
import json
from typing import Any, Dict, List, Optional, Set, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_GOLD_PATH = os.getenv("RAG_GOLD_PATH", os.path.join(PROJECT_ROOT, "data", "gold_dataset.json"))

# (source file name, 1-based page). Source is None for legacy items that only carry expected_pages.
PageKey = Tuple[Optional[str], int]


def load_gold(path: Optional[str] = None) -> Dict[str, Any]:
    with open(path or DEFAULT_GOLD_PATH, encoding="utf-8") as f:
        data = json.load(f)
    # A bare list of {question, expected_pages} is also accepted
    if isinstance(data, list):
        data = {"documents": [], "items": data}
    return data


def load_gold_items(path: Optional[str] = None) -> List[Dict[str, Any]]:
    return load_gold(path)["items"]


def gold_pages(item: Dict[str, Any]) -> Set[PageKey]:
    if "evidence" in item:
        return {(ev["source"], p) for ev in item["evidence"] for p in ev["pages"]}
    return {(None, p) for p in item.get("expected_pages", [])}


def item_sources(item: Dict[str, Any]) -> Set[str]:
    return {ev["source"] for ev in item.get("evidence", [])}


def is_answerable(item: Dict[str, Any]) -> bool:
    return bool(gold_pages(item))
