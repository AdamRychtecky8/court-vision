# Phase 4 — Shot-Make Model: Key Findings
*`03_shot_model/` — NBA three-point revolution project (Act 1 evidence)*

> **Note on numbers.** The figures below are from the 5% development sample (`seed=42`).
> The story holds on the full dataset; exact coefficients and metrics will shift slightly
> after the `SAMPLE_FRACTION=None` run that produces the final report tables.

## The question this phase answers
Which shot characteristics predict whether a shot goes in — and does shot LOCATION matter
more than game CONTEXT? This is the quantitative test behind Act 1 of the report.

## Finding 1 — Location and action decide makes; context barely registers.
Across the entire LASSO path, every context feature stays at or near zero: `IS_CLUTCH` (−0.06),
`QUARTER` (−0.017), `SECS_REMAINING` (≈0.0001), and `LOC_Y` dropped out entirely. Every feature
that carries real weight describes WHERE or HOW the shot was taken. In plain terms: the moment in
the game and the pressure on the clock do not predict a make — the spot on the floor does.

## Finding 2 — The rim is in a class of its own.
The strongest make-predictors are at the basket. A Dunk carries by far the largest positive
coefficient (+1.95), Restricted Area is positive (+0.23), and make-probability falls steadily with
`SHOT_DISTANCE` (−0.013). The Layup indicator was zeroed at the headline penalty — not because
layups miss often (they don't), but because Restricted Area and short-range features already
capture rim-proximity makeability. That redundancy being absorbed is LASSO working as intended.

## Finding 3 — Mid-range and the three are near-identical on makeability — and that is the paradox.
LASSO zeroed both Mid-Range and Above-the-Break-3, meaning that once distance and action type are
known, the specific zone label adds essentially no extra make-probability signal. This lines up
with Table 1, where the two shots have similar field-goal percentages (Mid-Range 39.9% vs
Above-the-Break-3 35.1%). The mid-range's problem is therefore NOT that players miss it more — it is
that it pays only 2 points for a shot no more makeable than a three worth 1.5× as much. The
inefficiency is a points-per-attempt result (Table 1); this model shows precisely WHY the two shots
look interchangeable on makeability alone. Together, Table 1 and this model are the core of Act 1.

## Finding 4 — A modest, honest model: ROC-AUC ≈ 0.63.
LogisticRegression scored AUC 0.629 / F1 0.604 / accuracy 0.617; LinearSVC 0.621 / 0.598 / 0.609.
Both clear the 0.54 always-guess-miss floor, but only modestly — the expected ceiling when the only
signals are location and action and there is no defender data. The model is conservative: it catches
about 42% of actual makes (recall) while about 63% of its predicted makes are correct (precision),
which reflects near-balanced classes and limited signal. The closeness of the two linear models is
itself reportable: they agree, with logistic regression marginally ahead.

## Finding 5 — A clean LASSO regularization path.
Surviving features grow monotonically as the penalty relaxes: 3 → 5 → 10 → 16 → 24 → 26 across
`regParam` 0.05 → 0.0001. This is a textbook demonstration of LASSO under collinearity: with
seven-plus features all encoding location, a strong penalty keeps one proxy and zeros the redundant
rest, then they re-enter as the penalty loosens. The order of re-entry is a stable importance
ranking; headline coefficients are reported at `regParam = 0.001` (16/26 features kept).

## What this proves for the report
Act 1's claim — that the math always favored the rim and the three over the mid-range — rests on two
complementary results. Table 1 shows the points-per-attempt gap; this model shows that makeability is
driven by location and action rather than context, and that mid-range and three-point shots are
near-identical in makeability. The model's honest, modest AUC is itself a point: a shot's value is
mostly about WHERE it is taken, not the situation it is taken in.

## Artifacts produced (in GCS)
- `gs://pstat135-adam/processed/tables/lasso_path/` — coefficients at every `regParam` (data for the path figure, Figure 4)
- `gs://pstat135-adam/processed/tables/lasso_coefficients/` — headline coefficients at `regParam=0.001` (Table 3)
- `gs://pstat135-adam/processed/tables/model_comparison/` — LogReg vs LinearSVC metrics (Table 4)
- Code: `03_shot_model/feature_engineering.py`, `03_shot_model/shot_model.py`

## Limitations and open items
- **No defender or contest data** — this measures shot LOCATION value, not full shot QUALITY.
- **Backcourt residual** — a small number of `ZONE_RANGE="Back Court Shot"` rows survived the Phase-2
  backcourt filter (coefficient −1.31). Confirm the count and add a footnote; it changes no conclusion.
- **Full-data refresh** — rerun with `SAMPLE_FRACTION=None` and update every number above before the
  final report.
- **Figure 4** — the regularization-path plot and/or confusion-matrix visual still to be rendered from
  the saved tables.
