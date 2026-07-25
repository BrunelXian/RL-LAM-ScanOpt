from __future__ import annotations


def block_interleaved_quarters(n: int) -> list[int]:
    starts = [round(i * n / 4) for i in range(4)]
    ends = starts[1:] + [n]
    blocks = [list(range(starts[i], ends[i])) for i in range(4)]
    order: list[int] = []
    max_len = max(len(block) for block in blocks)
    for offset in range(max_len):
        for block in blocks:
            if offset < len(block):
                order.append(block[offset])
    return order
