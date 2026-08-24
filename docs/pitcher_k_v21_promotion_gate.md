# Pitcher K V2.1 promotion gate

V2.1 is a research challenger only. V1 and V2 prospective histories remain frozen and unchanged.

Statcast-enhanced K features may be promoted only if all of the following hold in walk-forward testing:

1. Mean Poisson deviance is lower than the current K baseline.
2. At least 4 of 5 test seasons (2022-2026) show no material degradation in Poisson deviance.
3. Mean MAE is not worse by more than 0.01 strikeouts.
4. Statcast feature coverage is at least 90% of eligible starter rows in each evaluated season.
5. No feature uses current-game pitch data; all rolling Statcast features are shifted by one appearance before aggregation.

Even if these historical gates pass, V2.1 remains research-only until it accumulates its own prospective sample. Historical improvement alone does not replace V1/V2 prospective validation.
