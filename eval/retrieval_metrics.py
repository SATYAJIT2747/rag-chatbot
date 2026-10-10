import math
from typing import Hashable, List, Optional, Sequence, Set


def first_hit_rank(retrieved: Sequence[Hashable], gold: Set[Hashable], k: int) -> Optional[int]:
    for rank, key in enumerate(retrieved[:k], 1):
        if key in gold:
            return rank
    return None


def hit_at_k(retrieved: Sequence[Hashable], gold: Set[Hashable], k: int) -> float:
    return 1.0 if first_hit_rank(retrieved, gold, k) else 0.0


def recall_at_k(retrieved: Sequence[Hashable], gold: Set[Hashable], k: int) -> float:
    if not gold:
        return 0.0
    return len(gold & set(retrieved[:k])) / len(gold)


def mrr_at_k(retrieved: Sequence[Hashable], gold: Set[Hashable], k: int) -> float:
    rank = first_hit_rank(retrieved, gold, k)
    return 1.0 / rank if rank else 0.0


def ndcg_at_k(retrieved: Sequence[Hashable], gold: Set[Hashable], k: int) -> float:
    """Binary-relevance nDCG. A gold key only earns gain the first time it appears,
    so several chunks from the same gold page are not counted twice."""
    if not gold:
        return 0.0
    seen: Set[Hashable] = set()
    dcg = 0.0
    for rank, key in enumerate(retrieved[:k], 1):
        if key in gold and key not in seen:
            seen.add(key)
            dcg += 1.0 / math.log2(rank + 1)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold), k) + 1))
    return dcg / idcg


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0
