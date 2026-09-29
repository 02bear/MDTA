# Warm-start Rank16 + PCIM + standard pair-RNC + R1

This experiment uses the fixed DAVIS random-pair 8:1:1 split with seed 42.
The Rank16 backbone is trained from scratch on the warm-start training pairs.
PCIM uses bidirectional coverage Top-64 interaction and standard pair-level
Rank-N-Contrast with weight 0.01. R1 is adapted to sparse pair splits: every
query correction uses residuals from training pairs only. Test labels are
read only after validation selection is persisted.

Stages are resumable: `base`, `cache`, and `adapter`.
