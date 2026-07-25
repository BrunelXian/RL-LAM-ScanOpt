from __future__ import annotations


def decode_by_scores(scores: dict[int, float], legal_tracks: list[int]) -> list[int]:
    return sorted(legal_tracks, key=lambda x: (-scores.get(x, 0.0), x))


def legal_tracks_from_prefix(n: int, prefix: list[int] | None = None) -> list[int]:
    used = set(prefix or [])
    return [i for i in range(n) if i not in used]
