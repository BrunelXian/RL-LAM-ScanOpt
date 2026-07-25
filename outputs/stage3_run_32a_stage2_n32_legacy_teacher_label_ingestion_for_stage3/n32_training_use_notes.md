# N32 Training Use Notes

- N32 can be used as a legacy-compatible additional N group with warnings.
- U2 and SurfaceT mapping are stronger.
- PEEQ and Mises are proxy-compatible, not literal identical metric names.
- Some rows have missing mapped Mises proxy values; those rows are flagged and use conservative worst-observed proxy fill values only so rank/reward tables remain mechanically complete.
- Training should use per-N balancing or sample weighting because N32 has 332 rows versus N12/N16=32 and N24/N40=54.
- Claims should describe N32 as legacy-compatible Stage 2 32-track teacher data, not native Stage 3 newly generated teacher validation.
- N32 is valuable as an intermediate N between N24 and N40 for graph-policy generalization analysis.
