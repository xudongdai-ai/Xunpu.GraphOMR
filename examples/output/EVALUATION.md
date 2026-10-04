# Synthetic Stress Evaluation

Two procedural fixtures, eight variants each; not independent real-world scores.

Index-aligned matches / max(expected count, detected count, 1); macro average across cases.

| Case | Pitch | Duration | Legal measures | XML |
| --- | ---: | ---: | ---: | --- |
| melody-clean | 100.0% | 100.0% | 100.0% | True |
| melody-scale_075 | 100.0% | 100.0% | 100.0% | True |
| melody-scale_150 | 100.0% | 100.0% | 100.0% | True |
| melody-rotate_minus2 | 100.0% | 100.0% | 100.0% | True |
| melody-rotate_plus2 | 100.0% | 100.0% | 100.0% | True |
| melody-blur | 100.0% | 100.0% | 100.0% | True |
| melody-jpeg_45 | 100.0% | 100.0% | 100.0% | True |
| melody-noise_12 | 100.0% | 100.0% | 100.0% | True |
| durations-clean | 100.0% | 100.0% | 100.0% | True |
| durations-scale_075 | 100.0% | 33.3% | 50.0% | True |
| durations-scale_150 | 100.0% | 100.0% | 100.0% | True |
| durations-rotate_minus2 | 100.0% | 100.0% | 100.0% | True |
| durations-rotate_plus2 | 100.0% | 100.0% | 100.0% | True |
| durations-blur | 100.0% | 100.0% | 100.0% | True |
| durations-jpeg_45 | 100.0% | 100.0% | 100.0% | True |
| durations-noise_12 | 50.0% | 75.0% | 50.0% | True |

Limitations: no independent real-world evaluation, trained weights, or Audiveris superiority claim.
Timing includes recognition and artifact export; it is not isolated inference latency.

Reproduce: `xunpu-graph-omr portfolio --output out/evaluation`.
