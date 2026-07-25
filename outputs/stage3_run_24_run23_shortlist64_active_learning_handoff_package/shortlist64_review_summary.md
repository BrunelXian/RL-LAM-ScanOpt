# Shortlist64 Review Summary

- Total count: 64
- Per-N counts: {12: 8, 16: 8, 24: 24, 40: 24}
- N24/N40 share: 48/64
- Bucket composition: {'top_region_local_search': 14, 'model_disagreement': 10, 'uncertainty_calibration': 10, 'diversity_coverage': 10, 'tradeoff_probe': 8, 'sentinel_control': 6, 'exploitation_reference': 6}
- Candidate family composition: {'known_best_mutation': 50, 'combined108_seed_mutation': 2, 'teacher_best_local_search': 6, 'negative_control_low_adjacent': 1, 'geometry_signal_center_bias': 1, 'diversity_random': 1, 'geometry_signal_parity_switch': 3}
- Surrogate-only candidates predicted above combined108 best: 0
- Expected Abaqus cost: 64 jobs total, with 48 jobs from N24/N40.

Shortlist64 is selected because the user wants an overnight run with 60+ candidates. It gives stronger N24/N40 calibration coverage than batch32 or batch24.

This package is handoff only. No CAE/INP files were created, and no candidate is teacher-validated.
