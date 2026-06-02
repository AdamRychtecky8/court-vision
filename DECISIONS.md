Phase 1:

Actual column names differ from NBA Stats API standard: SHOT_MADE (boolean) not SHOT_MADE_FLAG (int), BASIC_ZONE not SHOT_ZONE_BASIC, ZONE_NAME not SHOT_ZONE_AREA. SEASON_1 exists as a clean integer so no date parsing needed. SHOT_MADE must be cast to integer before MLlib.

Phase 2:

## Phase 4 — Shot-make model

- **Leakage columns excluded from features:** EXPECTED_PTS (= SHOT_VALUE × SHOT_MADE_INT,
  so it equals 0 exactly when missed), EVENT_TYPE ("Made/Missed Shot"), and SHOT_MADE
  (boolean target). Including any would push AUC toward 1.0 — the all-clear was AUC ~0.63.
- **ACTION_TYPE grouped into 5 buckets** (Dunk, Layup, Hook, Jump Shot, Other) via keyword
  matching; raw ACTION_TYPE has dozens of values that would explode the one-hot width into
  sparse noise. "Other" is ~1.4% of shots.
- **Transformers fit on the train split only** and applied to test; StringIndexer uses
  handleInvalid="keep" so a category seen only in test is parked, not crashed. This is the
  no-leakage discipline.
- **LASSO shown as a regularization path** (regParam swept 0.05 → 0.0001), not a single
  value: at strong penalties, collinear location features collapse onto one proxy and hide
  the zone structure. Headline coefficients reported at regParam=0.001 (kept 16/26).
- **Model comparison:** LogisticRegression (light L2) vs LinearSVC; LogReg marginally better.
- **Interpretation note:** the make-model establishes location/action ≫ context for
  make-probability; the mid-range inefficiency comes from Table 1's points-per-attempt, not
  from this model. The two are complementary evidence for Act 1.
- **Limitation:** AUC ~0.63 reflects location/action-only signal — no defender/contest data.
- Dev work on 5% sample (seed=42); final tables produced with SAMPLE_FRACTION=None (full data).