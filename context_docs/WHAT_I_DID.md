# What I Did — court-vision (interview reference)

*Private study aid. The goal isn't to memorize lines — it's to be able to explain the **why** behind every choice, because that's what interviewers probe. Read the decisions and concept sections until you can say them in your own words without notes.*

---

## 30-second pitch (have this ready)

"I built an end-to-end big-data pipeline on Google Cloud that analyzes 4.2 million NBA shots across 21 seasons to quantify the three-point revolution. I ingested raw CSVs into Cloud Storage, processed and feature-engineered them with PySpark on a Dataproc cluster, stored the result as season-partitioned Parquet, and ran three MLlib analyses: a LASSO-selected shot-make classifier, a season-level era time series, and a PCA-plus-K-Means clustering of player shot profiles. The headline finding is that the mid-range is mathematically dominated by the three, the league tipped in 2015–2017, and the mid-range-specialist player archetype collapsed from 37% to 8% of the league afterward."

---

## What the system actually does (architecture)

A four-stage pipeline, all on GCP:

1. **Ingestion** — 21 season CSVs uploaded to a GCS bucket (`raw/`); a verification job confirmed 4,231,262 rows, 26 columns, zero nulls.
2. **Processing** — a PySpark job on Dataproc removed 8,820 backcourt heaves (→ 4,222,442 rows), added 7 engineered columns, and wrote the result as **Parquet partitioned by season** to `processed/`.
3. **Modeling** — three Spark MLlib analyses read the Parquet and wrote small result tables back to GCS.
4. **Reporting** — a Quarto document reads those small tables + matplotlib figures and renders a PDF. Compute lives on the cluster; the report is a thin presentation layer.

The whole thing is reproducible: fixed seeds, documented run order, datasets in GCS (never in Git), decisions logged in `DECISIONS.md`.

---

## Data engineering (be able to speak to this — it's the most job-relevant part)

- **Scale & format.** Raw CSVs → GCS → Parquet. The honest **Spark justification**: a single season fits on a laptop; what warrants distributed processing is the *cross-era* work — full-panel aggregations, a classifier trained on all 4.2M rows, and PCA/K-Means over every player-season treated as one dataset.
- **Why Parquet over CSV.** Columnar storage means you read only the columns you need (column pruning), it's compressed (Snappy), and row-group statistics enable predicate pushdown. **Partitioning by season** lets Spark skip whole files when filtering to a season (partition pruning).
- **Lazy evaluation & caching.** Spark transformations (`select`, `groupBy`, the ML transformers) just build a logical plan; nothing runs until an **action** (`count`, `show`, `write`, `collect`, fitting a model). I `.cache()`d DataFrames I hit with multiple actions so Spark didn't re-read Parquet and recompute each time.
- **Feature engineering.** 7 derived columns, including `SECS_REMAINING` (clock as one integer), `IS_CLUTCH` (Q4/OT, ≤120s), `SHOT_MADE_INT` (integer cast for MLlib), and binary shot-value flags.
- **Aggregation patterns.** Player-season profiles were built with a **pivot** (`groupBy(player, season).pivot(zone).count()`) then converted to zone *shares* — collapsing 4.2M shot rows into ~6,451 player-season rows.
- **Cluster lifecycle / cost.** Single-node `e2-highmem-4`; started before each job, **stopped after every session** because it bills while running. An IAM fix (granting the compute service account `dataproc.worker`) was needed to submit jobs.

---

## The machine learning (three analyses)

**1. Shot-make classifier (supervised).**
- Pipeline: `StringIndexer` → `OneHotEncoder` → `VectorAssembler` (26-dim vector: 8 numeric + 4 one-hot categoricals).
- **LASSO** (`LogisticRegression`, `elasticNetParam=1.0`) for feature selection, then full **LogisticRegression** and **LinearSVC** classifiers.
- Evaluated with `BinaryClassificationEvaluator` (ROC-AUC) and `MulticlassClassificationEvaluator` (F1, accuracy) + a confusion matrix on an 80/20 split (seed 42).
- Result: AUC ≈ 0.63; location/action features dominate, context features zero out.

**2. Era analysis (descriptive).**
- Three-point attempt rate aggregated by season, 2004–2024; rendered as a time series. Showed the flat ~22% decade and the 2015–2017 acceleration.

**3. Player archetypes (unsupervised).**
- `StandardScaler` → `PCA` (5 of 6 components) → `K-Means` (k=5, seed 42) on six-dimensional zone-share profiles.
- Held clusters fixed and compared pre/post-2015 populations to show role reorganization.

---

## Core concepts I must be able to explain

**Data leakage.** Three columns were target-derived and excluded: `EXPECTED_PTS` (= shot value × made, so it's 0 exactly when missed), `EVENT_TYPE` ("Made/Missed Shot"), `SHOT_MADE` (the boolean target). I also fit all transformers on the **train split only** then applied them to test. The check that it worked: AUC came out ~0.63, not ~1.0 — a near-perfect score would have meant a leaked feature.

**One-hot encoding.** Categories → numbers via `StringIndexer`, then → binary vectors via `OneHotEncoder` so the model doesn't read index 5 as "greater than" index 4 (the numbers are just labels, like jersey numbers). `handleInvalid="keep"` parks a category seen only in test in a reserved bucket instead of crashing.

**L1 (LASSO) vs L2 (Ridge).** L1 drives weak coefficients to *exactly zero* (automatic feature selection); L2 shrinks but keeps everything. **Under collinearity**, LASSO keeps one of a correlated group and zeros the rest — which is why I reported a **regularization path** (swept `regParam` 0.05→0.0001, watched the active-feature count go 3→5→9→16→21→26) instead of one penalty value: the *order* features re-enter is a more stable importance ranking. I standardized features first so units don't bias which ones get penalized.

**Why AUC/F1 over accuracy.** Classes are near 46/54 (made/missed), so the always-guess-miss accuracy floor is ~54%. AUC measures ranking quality independent of the decision threshold; F1 balances precision and recall. Reporting accuracy alone would have hidden how the model performs per class.

**PCA.** Rotates correlated features into uncorrelated axes ordered by variance. The **loadings** tell you what each axis means — PC1 (51.5% of variance) loaded negative on rim/paint and positive on threes, so it's an "interior ↔ perimeter" axis; PC2 (23%) was "mid-range ↔ rim." **Compositional data:** the six zone shares sum to 1, so one direction carries near-zero real variation — that 6th component was dropped. I standardized before PCA so high-variance zones didn't dominate.

**K-Means & choosing k.** Partitions points into k clusters minimizing within-cluster variance. I scored k=3..8 by **silhouette** — k=4 edged k=5 (0.411 vs 0.376) — but chose **k=5** because it produced five clearly nameable basketball roles while the marginal silhouette gain at k=4 came from merging genuinely distinct roles. (Good interview point: I let *interpretability* override a marginal metric, and documented why.)

**MLlib Pipeline.** Chains transformers + model into one object so the exact same fitted transforms apply to train and test — structurally prevents train/test skew.

---

## Decisions & tradeoffs (this is the interview gold)

- **Parquet, partitioned by season** — columnar + compressed + partition pruning for the season-level analyses.
- **LASSO over plain logistic** — wanted feature *selection* and an interpretable importance ranking, not just prediction.
- **Regularization path over a single penalty** — discovered that a strong penalty collapsed collinear location features onto one proxy and hid the zone story; the path is robust to that.
- **k=5 despite a marginally better silhouette at k=4** — interpretability over a small metric gain.
- **Dev on a 5% sample, final on full data** — fast iteration while building, full ~4.2M-row run for the reported numbers.
- **Single-node cluster** — a deliberate cost/scale tradeoff for a student project; I'd scale to multi-node for genuinely larger data.

---

## Results to know cold

- **4,222,442** shots, **21** seasons (2004–2024).
- Mid-range **0.798** pts/attempt vs above-the-break-3 **1.054** — a **32%** gap at similar FG% (39.9% vs 35.1%).
- Shot-make model AUC **0.63**; context features zeroed.
- 3PT rate **18.6% → 39.4%**; sharpest jump 2016→2017.
- Five archetypes; mid-range specialist **37.3% → 7.7%**, modern balanced **8.1% → 35.8%**.

---

## Likely questions + how to answer

- **"Why Spark / when would you not use it?"** — Cross-era full-panel work justifies it; a single season doesn't. Knowing the boundary is the point.
- **"How did you prevent leakage?"** — Excluded target-derived columns; fit transforms on train only; AUC ~0.63 (not ~1.0) was the sanity check.
- **"Most interesting problem you hit?"** — The LASSO mass zero-out: a strong penalty zeroed 21 of 26 features and hid the zone story. I recognized it as LASSO-under-collinearity, switched to a regularization path, and got a robust importance ranking — turning a confusing result into a stronger one.
- **"What would you do next?"** — Add defender/contest data (the model's biggest limitation), build the GraphFrames assist-network analysis, add a disaggregated fairness/robustness audit, and tune hyperparameters with cross-validation instead of fixed values.

---

## Honest framing

This was AI-assisted (I used Claude for planning, code drafting, and explanation), and I drove the architecture, made every analytical decision, interpreted the results, and reviewed all the code before running it. In an interview, lean on the parts you genuinely own: the *system design*, the *decisions and tradeoffs* above, the *concepts*, and the *interpretation* of the findings. If asked to go deeper on a piece of code, be honest that you can reason about what it does and why — that's more credible than overclaiming line-by-line authorship.
