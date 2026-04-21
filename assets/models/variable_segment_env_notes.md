# Variable Segment Environment Notes

New variable-length segment environment
- Base class: ScanPlanningSegmentEnv
- Historical fixed action: (stripe_id, segment_index) with balanced fixed split into 6 parts
- New variable-length action: (stripe_id, start_cell, length) where length is chosen from 2 to 8
- Execution semantics: one action scans the currently unscanned cells inside that stripe window
- Reward logic, thermal proxy, observation layout, and variant_1 reward weights are unchanged
