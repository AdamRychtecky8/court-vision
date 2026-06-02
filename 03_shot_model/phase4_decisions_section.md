## Phase 4 — Shot-make model

**Goal.** Build a binary make/miss classifier and use LASSO to identify which shot
characteristics predict a make — the evidence for Act 1 ("location dominates context").

- **Target.** `SHOT_MADE_INT` (1 = made, 0 = missed). League base rate ≈ 46% makes, so the
  always-guess-miss accuracy floor is ≈ 54%; the model is judged on ROC-AUC and F1, not accuracy.

- **Leakage columns excluded from features.** `EXPECTED_PTS` (= `SHOT_VALUE` × `SHOT_MADE_INT`,
  so it equals 0 exactly when a shot is missed), `EVENT_TYPE` ("Made Shot"/"Missed Shot"), and
  `SHOT_MADE` (boolean target) are all derived from or equal to the label and were kept out of
  the feature vector. The resulting AUC of ~0.63 (not ~1.0) is the confirmation that nothing leaked.

- **Feature set (assembled vector length = 26).** 8 numeric, used as-is: `LOC_X`, `LOC_Y`,
  `SHOT_DISTANCE`, `SECS_REMAINING`, `IS_3PT`, `SHOT_VALUE`, `IS_CLUTCH`, `QUARTER`. 4 categorical,
  indexed then one-hot encoded: `BASIC_ZONE`, `SHOT_TYPE`, `ZONE_RANGE`, `ACTION_GROUP`.

- **`ACTION_TYPE` → `ACTION_GROUP`.** The dozens of raw action values were collapsed into 5
  buckets — Dunk, Layup, Hook, Jump Shot, Other — by keyword matching, to keep the one-hot width
  manageable instead of exploding into sparse noise. "Other" is ≈ 1.4% of shots.

- **No-leakage pipeline discipline.** Feature transformers (`StringIndexer` / `OneHotEncoder`)
  are fit on the TRAIN split only, then applied to the test split. `StringIndexer` uses
  `handleInvalid="keep"` so a category appearing only in the test split is parked in an extra
  bucket rather than crashing the job.

- **LASSO presented as a regularization path, not a single penalty.** At strong penalties
  (`regParam` ≥ 0.01) LASSO collapses the many collinear location features onto a single proxy and
  hides the zone-by-zone structure. We swept `regParam` = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
  (surviving features 3 → 5 → 10 → 16 → 24 → 26) and report headline coefficients at
  `regParam = 0.001` (kept 16/26). The path is itself a report figure; the order in which features
  re-enter is a collinearity-robust importance ranking.

- **Classifiers.** `LogisticRegression` (light L2, `regParam=0.01`) and `LinearSVC` (`regParam=0.01`),
  same feature vector, evaluated on a held-out 20% test split (`seed=42`). The two linear models
  agree closely; logistic regression is marginally ahead.

- **Interpretation guardrail (make-probability vs efficiency).** The make-model measures
  make-PROBABILITY, in which mid-range and three-point shots are similar (both near baseline). The
  mid-range's INEFFICIENCY is a points-per-attempt result (Table 1), not a make-model result. The
  two findings are complementary, not redundant — do not read "mid-range is bad" off the model
  coefficients.

- **Limitation.** AUC ~0.63 is the honest ceiling for location/action-only data; the dataset
  contains no defender or contest information, so this measures shot LOCATION value, not full
  shot QUALITY.

- **Sampling.** Developed on a 5% sample (`seed=42`); final report tables (3, 3a, 4) are produced
  with `SAMPLE_FRACTION=None` over all ~4.2M shots.

- **Open data-quality check.** A small residual of `ZONE_RANGE = "Back Court Shot"` rows survived
  the Phase-2 backcourt filter (it carries a strong negative coefficient, −1.31). Confirm the count
  and footnote it; it does not affect any conclusion.
