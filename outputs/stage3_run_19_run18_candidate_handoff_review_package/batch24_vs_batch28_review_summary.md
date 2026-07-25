# Batch24 vs Batch28 Review Summary

- Batch24 total: `24`; N24/N40 share: `18/24`.
- Batch28 total: `28`; N24/N40 share: `20/28`.
- Batch24 is cheaper and more conservative.
- Batch28 gives slightly broader N24/N40 coverage at the cost of 4 additional future jobs.
- Neither option is teacher-validated.
- Neither option should be treated as a guaranteed physical improvement over combined80 best cases.

| Batch | N | Count | Mean Pred Reward | Max Pred Reward | Mean Novelty | Mean Uncertainty | Above Combined80 Best? |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| batch24 | 12 | 3 | 0.6682 | 0.7768 | 0.1010 | 0.0812 | False |
| batch24 | 16 | 3 | 0.6771 | 0.7838 | 0.1083 | 0.0615 | False |
| batch24 | 24 | 9 | 0.6478 | 0.8347 | 0.2403 | 0.1345 | False |
| batch24 | 40 | 9 | 0.6543 | 0.8515 | 0.2595 | 0.1106 | False |
| batch28 | 12 | 4 | 0.6856 | 0.7768 | 0.0795 | 0.0777 | False |
| batch28 | 16 | 4 | 0.7038 | 0.7838 | 0.0958 | 0.0519 | False |
| batch28 | 24 | 10 | 0.6664 | 0.8347 | 0.2228 | 0.1268 | False |
| batch28 | 40 | 10 | 0.6732 | 0.8515 | 0.2433 | 0.1042 | False |

Review position: do not recommend either option as universally superior. Choose batch24 for a lower-cost calibration pass, or batch28 for broader N24/N40 coverage.
