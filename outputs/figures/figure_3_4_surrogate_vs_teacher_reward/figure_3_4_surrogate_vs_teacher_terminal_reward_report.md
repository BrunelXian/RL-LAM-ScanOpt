# Figure 3.4 Surrogate vs Teacher Terminal Reward Report

- Input data path: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\surrogate_reward_model\tables\surrogate_reward_model_candidate_predictions_holdout.csv`
- Number of validation cases used: `138`
- N-wise counts: `{'N12': 20, 'N16': 20, 'N24': 47, 'N40': 51}`
- Spearman correlation: `0.8786`
- Pearson correlation: `0.8863`
- Computed Spearman from plot data: `0.878558039394`
- Computed Pearson from plot data: `0.886310761234`
- PNG: `E:\Projects\RL-LAM-ScanOpt\outputs\figures\figure_3_4_surrogate_vs_teacher_reward\figure_3_4_surrogate_vs_teacher_terminal_reward.png`
- PDF: `E:\Projects\RL-LAM-ScanOpt\outputs\figures\figure_3_4_surrogate_vs_teacher_reward\figure_3_4_surrogate_vs_teacher_terminal_reward.pdf`
- SVG: `E:\Projects\RL-LAM-ScanOpt\outputs\figures\figure_3_4_surrogate_vs_teacher_reward\figure_3_4_surrogate_vs_teacher_terminal_reward.svg`
- Plot data CSV: `E:\Projects\RL-LAM-ScanOpt\outputs\figures\figure_3_4_surrogate_vs_teacher_reward\figure_3_4_surrogate_vs_teacher_terminal_reward_plot_data.csv`

Interpretation boundary: this figure validates the surrogate reward model as a PPO training environment. It is not a substitute for final Abaqus teacher validation, and final physical claims remain grounded in independent teacher simulations.
