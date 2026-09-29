# DAVIS warm-start low-cost tuning

This experiment freezes the completed Rank16 backbone and reuses its audited
feature cache.  It tunes only the lightweight PCIM/RNC adapter and the
validation-only lambda/R1 calibration.

Stages:

1. Expanded lambda/R1 calibration of the completed adapter.
2. PCIM Top-K, coverage-budget and width screening.
3. Standard RNC weight and temperature screening.
4. Three adapter seeds, locked validation selection, then one final test read.

The test labels are masked during all search stages.  The final action writes
`selection_locked.json` before evaluating test metrics and verifies that
poisoning validation/test labels cannot alter R1 predictions.
