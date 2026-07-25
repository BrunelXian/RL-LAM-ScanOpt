# Figure 3.x - PPO Candidate Surrogate-to-Teacher Alignment

## Input
- Alignment data: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\stageH_teacher_metric_ranking\tables\ppo_surrogate_vs_teacher_alignment.csv`
- Stage H summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\stageH_teacher_metric_ranking\tables\ppo_surrogate_vs_teacher_alignment_summary.json`

## Scope
- Candidates used: 32
- N-wise counts: N12=8, N16=8, N24=8, N40=8
- Data scope: 32 teacher-validated PPO-generated candidates only.

## Columns
- Surrogate-predicted terminal reward: `predicted_surrogate_reward_lex`
- Abaqus teacher-derived terminal reward: `teacher_lex_reward_rank_normalized`

## Correlations
- Spearman correlation: 0.2790
- Pearson correlation: 0.2092

## Generated Outputs
- PNG: `E:\Projects\RL-LAM-ScanOpt\outputs\figures\figure_3_x_ppo_candidate_surrogate_teacher_alignment\figure_3_x_ppo_candidate_surrogate_teacher_alignment.png`
- PDF: `E:\Projects\RL-LAM-ScanOpt\outputs\figures\figure_3_x_ppo_candidate_surrogate_teacher_alignment\figure_3_x_ppo_candidate_surrogate_teacher_alignment.pdf`
- SVG: `E:\Projects\RL-LAM-ScanOpt\outputs\figures\figure_3_x_ppo_candidate_surrogate_teacher_alignment\figure_3_x_ppo_candidate_surrogate_teacher_alignment.svg`
- Plot data: `E:\Projects\RL-LAM-ScanOpt\outputs\figures\figure_3_x_ppo_candidate_surrogate_teacher_alignment\figure_3_x_ppo_candidate_surrogate_teacher_alignment_plot_data.csv`

## Interpretation
The weak but positive surrogate-to-teacher alignment indicates that the surrogate provides a useful prioritisation signal for PPO-generated candidates, but final physical evaluation must remain grounded in independent Abaqus teacher validation.
