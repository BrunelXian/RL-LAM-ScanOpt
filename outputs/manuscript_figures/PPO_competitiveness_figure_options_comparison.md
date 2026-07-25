# PPO Competitiveness Figure Options

## Shared Data Boundary

All three figures use the same native-N PPO competitiveness summary:

| N | Native reference pool | PPO candidates | Top-k competitive candidates | Best PPO rank | Best-rank percentile |
|---|---:|---:|---:|---:|---:|
| N12 | 78 | 8 | 5 | 6 | 7.7% |
| N16 | 78 | 8 | 4 | 2 | 2.6% |
| N24 | 190 | 8 | 3 | 134 | 70.5% |
| N40 | 206 | 8 | 0 | 147 | 71.4% |

N32, N50, and N64 are excluded from this ranking boundary. All rankings are relative to the native reference pool for the same N.

## Figure A: Competitiveness Profile Cards

Best use: supplementary figure or a compact evidence-summary graphic.

Strengths:
- Most explicit: every number needed for interpretation is printed.
- Useful for readers who want the exact pool size, candidate count, top-k count, rank, and percentile.
- Clearly communicates the bounded claim without needing much caption support.

Limitations:
- More table-like than figure-like.
- Less elegant as a main Results figure if space is tight.

## Figure B: Best-Rank Percentile Bar

Best use: Discussion section or a simple supporting panel.

Strengths:
- Very direct at showing the small-N versus large-N split.
- Easy to read quickly.
- Strong visual emphasis on "lower is better" and N12/N16 competitiveness.

Limitations:
- Shows only the best candidate per N.
- Does not communicate top-k occupancy among the 8 PPO candidates.

## Figure C: Bubble Competitiveness Plot

Best use: main manuscript figure.

Strengths:
- Combines best-candidate competitiveness and top-k occupancy in one compact view.
- Makes bounded competitiveness clear: N12/N16 strong, N24 limited, N40 absent in top-k occupancy.
- Best single figure for Results because it conveys both rank quality and cohort-level occupancy.

Limitations:
- Requires a legend explaining marker size.
- Exact top-k counts are less explicit than in Figure A, though still visually encoded.

## Recommendation

If only one figure can be retained, use **Figure C: Joint view of PPO competitiveness**. It is the strongest manuscript-facing option because it combines the best-rank percentile and the number of top-k competitive PPO candidates, making the bounded competitiveness claim visually clear without implying all-N dominance.

Suggested placement:
- Main Results: Figure C.
- Supplementary: Figure A.
- Discussion or compact narrative support: Figure B.
