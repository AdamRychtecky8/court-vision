# court-vision — Project Context for New Conversation
## Paste this at the start of a new chat to continue where we left off

---

## What this is

This document gives a new AI assistant full context to continue helping me build
my PSTAT 135 final project. Read all sections before responding. The project is
in active development — Phases 0, 1, and 2 are complete; Phase 3 starts now.

---

## Project Overview

**Course:** PSTAT 135 — Big Data Analytics, Prof. Ichiba, UCSB, Spring 2026
**Deliverable:** ~10-page written report (Word/RMarkdown/LaTeX) + short GauchoCast
video presentation, submitted during finals week (~June 14-20, 2026).
**GitHub repo:** court-vision (private, will go public after submission)

**The story:** The NBA underwent a three-point revolution. This project quantifies
why it was mathematically inevitable, when exactly it happened, and what it did to
the kinds of players who succeed — told in three connected analytical acts:

- **Act 1 (Phase 3):** Shot-make prediction — LASSO for feature selection,
  then classification — proves location dominates and mid-range is inefficient.
- **Act 2 (Phase 4):** Era analysis — 3PT rate by season shows the inflection at
  2015-2017, aligned with the Warriors dynasty.
- **Act 3 (Phase 5):** Player archetypes — PCA + K-Means shows player roles
  reorganized around the three-point line post-2015.
- **Phase 6 (optional):** Fairness/robustness audit — disaggregate model performance
  by era and zone (Lecture 16).
- **Phase 7 (future/appendix only):** GraphFrames assist network — not needed for
  the grade, not needed now.

The grade is on the **report and video**, not the codebase.

---

## Current State — What Is Done

### Phase 0 — Repo + Environment ✅
- GitHub repo `court-vision` created and pushed (private)
- Folder structure created: 01_ingestion through 07_evaluation, notebooks, data, results, report
- .gitignore excludes data/, *.csv, *.parquet, *.json credentials, __pycache__
- GCP fully configured (see Technical Environment below)
- Cluster smoke test passed

### Phase 1 — Ingestion ✅
- 21 CSVs (NBA_2004_Shots.csv through NBA_2024_Shots.csv) uploaded to GCS
- verify_shots.py ran successfully: 4,231,262 rows, 26 columns, 21 seasons, 0 nulls
- All ingestion docs committed to 01_ingestion/

### Phase 2 — Processing & Feature Engineering ✅
- process_shots.py ran successfully
- 8,820 backcourt shots removed → 4,222,442 rows in Parquet
- 7 new feature columns added (see Schema section)
- Parquet written to GCS, partitioned by SEASON_1
- Two analytical output tables produced and saved to GCS (see Key Findings)

### Phase 3 — Shot Model 🔴 STARTING NOW
- Folder: 03_shot_model/ (empty)
- Goal: LASSO feature selection + LogisticRegression + LinearSVC
- Read from: gs://pstat135-adam/processed/shots/ (Parquet)

---

## Technical Environment

### GCP
| Setting | Value |
|---------|-------|
| Project ID | pstat135-hw-497220 |
| Project display name | court-vision |
| Bucket | pstat135-adam (US multi-region — compatible with us-central1) |
| Cluster name | mycluster |
| Cluster type | Single Node |
| Machine | e2-highmem-4 |
| Region | us-central1 |
| Zone | us-central1-a |
| Dataproc image | 2.2.81-debian12 |
| Optional components | JUPYTER + component gateway |
| IAM fix applied | serviceAccount:171975187367-compute@developer.gserviceaccount.com → roles/dataproc.worker |
| Staging bucket | dataproc-staging-us-central1-171975187367-xo2mj2uu |
| Temp bucket | dataproc-temp-us-central1-171975187367-apjjbpx7 |

### gcloud defaults set
```
project     = pstat135-hw-497220
compute/region = us-central1
compute/zone   = us-central1-a
dataproc/region = us-central1
```

### Local machine
- Windows, PowerShell (NOT bash)
- Use backtick `` ` `` for line continuation, NOT backslash
- No `touch` (use `New-Item`), `&&` works only in PS7+ (use separate lines to be safe)
- Python 3.13.5, Git 2.48.1.windows.1, gcloud SDK 570.0.0
- VSCode with Python + Jupyter extensions; all commands run in integrated terminal

### Standard job submit pattern (from court-vision/ root)
```powershell
gcloud dataproc clusters start mycluster
gcloud dataproc jobs submit pyspark SCRIPT_PATH.py --cluster=mycluster
gcloud dataproc clusters stop mycluster
```

**Stop the cluster after every session — it costs ~$0.22/hr while running.**

### Reading job output
Output streams to the terminal during the job. If it doesn't appear inline,
the output files are at:
`gs://dataproc-staging-us-central1-171975187367-xo2mj2uu/google-cloud-dataproc-metainfo/
3c4c33cd-cfa3-417d-a938-f995fa6ae977/jobs/<JOB_ID>/driveroutput.000000000`

---

## Repo File Structure

```
court-vision/
├── README.md
├── DECISIONS.md
├── requirements.txt
├── .gitignore
├── smoke_test.py
├── debug_test.py
├── 01_ingestion/
│   ├── setup_commands.md     ← full GCP command reference
│   ├── ingest_shots.md       ← data source + upload + verification record
│   └── verify_shots.py       ← Phase 1 verification job
├── 02_processing/
│   └── process_shots.py      ← Phase 2 processing + feature engineering
├── 03_shot_model/            ← EMPTY — Phase 3 starts here
├── 04_era_analysis/          ← empty
├── 05_clustering/            ← empty
├── 06_networks/              ← empty (optional GraphFrames)
├── 07_evaluation/            ← empty
├── notebooks/                ← empty
├── data/
│   └── README.md             ← data dictionary (all 26 original + 7 new columns)
├── results/
│   └── key_findings.md       ← actual findings with numbers
└── report/                   ← empty
```

---

## Data State

### GCS paths
```
gs://pstat135-adam/raw/shots/NBA_*_Shots.csv       ← raw input (21 files, never touched again)
gs://pstat135-adam/processed/shots/                ← Parquet, partitioned by SEASON_1 ← ALL PHASES READ FROM HERE
gs://pstat135-adam/processed/tables/fg_pct_by_zone/           ← Table 1 CSV
gs://pstat135-adam/processed/tables/three_pt_rate_by_season/  ← Table 2 CSV
```

### Parquet schema — all 33 columns
**CRITICAL: column names differ from the NBA Stats API standard.**
Do not use SHOT_MADE_FLAG, SHOT_ZONE_BASIC, SHOT_ZONE_AREA, SHOT_ZONE_RANGE — they don't exist.

| Column | Type | Notes |
|--------|------|-------|
| SEASON_1 | integer | Season year (2004-2024) — PRIMARY season identifier |
| SEASON_2 | string | Formatted "2021-22" — label only |
| TEAM_ID | integer | |
| TEAM_NAME | string | |
| PLAYER_ID | integer | Join key |
| PLAYER_NAME | string | |
| POSITION_GROUP | string | G, F, or C |
| POSITION | string | e.g. "SG-PG" |
| GAME_DATE | string | MM-DD-YYYY format |
| GAME_DATE_PARSED | date | Parsed version of GAME_DATE ← new in Phase 2 |
| GAME_ID | integer | |
| HOME_TEAM | string | Abbreviation |
| AWAY_TEAM | string | Abbreviation |
| EVENT_TYPE | string | "Made Shot" or "Missed Shot" |
| SHOT_MADE | boolean | true/false ← original |
| SHOT_MADE_INT | integer | 1/0 cast of SHOT_MADE ← new in Phase 2; USE THIS for MLlib |
| ACTION_TYPE | string | Many values (Jump Shot, Layup, Dunk, etc.) — needs grouping for one-hot |
| SHOT_TYPE | string | "2PT Field Goal" or "3PT Field Goal" |
| BASIC_ZONE | string | Zone category ← USE THIS, not SHOT_ZONE_BASIC |
| ZONE_NAME | string | Sub-zone within BASIC_ZONE |
| ZONE_ABB | string | Zone abbreviation |
| ZONE_RANGE | string | Distance band |
| LOC_X | double | Horizontal court coordinate |
| LOC_Y | double | Vertical court coordinate |
| SHOT_DISTANCE | integer | Feet |
| QUARTER | integer | 1-4, 5+ for OT |
| MINS_LEFT | integer | Minutes remaining in quarter |
| SECS_LEFT | integer | Seconds remaining in minute |
| SECS_REMAINING | integer | MINS_LEFT×60 + SECS_LEFT ← new in Phase 2 |
| IS_3PT | integer | 1 if 3PT, 0 if 2PT ← new in Phase 2 |
| SHOT_VALUE | integer | 2 or 3 ← new in Phase 2 |
| IS_CLUTCH | integer | 1 if Q4/OT with ≤120 secs left ← new in Phase 2 |
| EXPECTED_PTS | integer | SHOT_VALUE × SHOT_MADE_INT (decimal when averaged) ← new in Phase 2 |

### BASIC_ZONE values (6 categories, backcourt removed)
Restricted Area, In The Paint (Non-RA), Mid-Range,
Left Corner 3, Right Corner 3, Above the Break 3

---

## Key Findings Already Produced (go into the report)

### Table 1 — Zone Efficiency (report Section 4.1)
| Zone | Attempts | FG% | Pts/Attempt |
|------|----------|-----|-------------|
| Restricted Area | 1,344,005 | 61.6% | 1.232 |
| Right Corner 3 | 146,724 | 38.8% | 1.164 |
| Left Corner 3 | 159,656 | 38.6% | 1.159 |
| Above the Break 3 | 890,037 | 35.1% | 1.054 |
| In The Paint (Non-RA) | 644,179 | 40.9% | 0.818 |
| Mid-Range | 1,037,841 | 39.9% | 0.798 |

**The finding:** Mid-Range and Above the Break 3 have nearly identical FG%
(39.9% vs 35.1%) but the 3-pointer generates 32% more points per attempt
(1.054 vs 0.798). Corner 3s are the second most efficient shot after the rim.

### Table 2 — 3PT Rate by Season (report Section 4.2 / Figure 2 data)
| Season | Total Attempts | 3PT Attempts | 3PT Rate | FG% |
|--------|---------------|--------------|----------|-----|
| 2004 | 189,467 | 35,159 | 18.6% | 43.9% |
| 2008 | 200,033 | 44,080 | 22.0% | 45.8% |
| 2012 | 160,901 | 36,071 | 22.4% | 44.9% |
| 2015 | 205,123 | 54,690 | 26.7% | 45.0% |
| 2016 | 207,453 | 58,645 | 28.3% | 45.3% |
| 2017 | 209,430 | 65,739 | 31.4% | 45.8% |
| 2019 | 218,992 | 78,276 | 35.7% | 46.1% |
| 2021 | 190,674 | 74,513 | 39.1% | 46.7% |
| 2024 | 218,268 | 85,922 | 39.4% | 47.5% |

**The finding:** Rate sat ~22% for a decade (2008-2014), then accelerated sharply
in 2015-2017 (Warriors dynasty / Curry unanimous MVP), plateauing ~38-40%
from 2020 onward. 2012 dip = lockout season (66 games, expected). 2020 dip = COVID bubble.

---

## Decisions Made (in DECISIONS.md)

- Column names differ from NBA API standard — corrected names documented above
- SHOT_MADE is boolean → SHOT_MADE_INT created as integer cast for MLlib
- Removed 8,820 backcourt shots (desperation heaves, not real attempts)
- Only the shot dataset is needed — no play-by-play required for core analyses
- Play-by-play (assist network, GraphFrames) is future-work/appendix only
- pstat135-adam bucket (US multi-region) works with us-central1 cluster — no cross-region charges
- SEASON_1 (integer) is the primary season identifier — no date parsing needed
- EXPECTED_PTS stored as integer (0, 2, or 3 per shot) — correct when averaged in aggregations

---

## Phase 3 — What Needs to Happen

**Goal:** Build a shot-make prediction model. Use LASSO (L1 penalty) for feature
selection to identify which shot characteristics most predict a make. Then train a
full classifier and evaluate it.

**Methods (from course Lectures 7 and 8):**
- Feature engineering: StringIndexer + OneHotEncoder for categorical columns,
  VectorAssembler to combine everything into one feature vector
- MLlib Pipeline to chain transformers + model in one object
- LASSO via LogisticRegression(elasticNetParam=1.0, regParam=...) — pure L1 penalty
- Classifier: LogisticRegression + LinearSVC
- Train/test split: fixed seed for reproducibility (e.g. seed=42)
- Evaluation: BinaryClassificationEvaluator (ROC-AUC),
  MulticlassClassificationEvaluator (F1, accuracy)
- Confusion matrix from predictions DataFrame

**Features for the model:**
Numeric (use as-is): LOC_X, LOC_Y, SHOT_DISTANCE, SECS_REMAINING, IS_3PT,
SHOT_VALUE, IS_CLUTCH, QUARTER
Categorical (need StringIndexer + OneHotEncoder): BASIC_ZONE, SHOT_TYPE,
ZONE_RANGE, ACTION_TYPE (ACTION_TYPE has many values — group into major
categories first: Dunk, Layup, Jump Shot, Post-Up, Other)
Target: SHOT_MADE_INT

**Output files for 03_shot_model/:**
- feature_engineering.py — builds the feature vector
- shot_model.py — LASSO feature selection + full classifier
- Results printed to terminal + saved to gs://pstat135-adam/processed/tables/

**Key report outputs from Phase 3:**
- Table 3: LASSO feature coefficients ranked by magnitude
- Table 4: Model comparison (LogReg vs LinearSVC) — ROC-AUC, F1, accuracy
- Figure 4: ROC curve and/or confusion matrix

---

## Working Style — How to Help

**Code approach:** Adam is not a CS major. He learns by reading well-commented
code rather than writing it first. Write complete scripts with thorough comments
explaining WHY each section exists, not just what it does. Use the same style as
verify_shots.py and process_shots.py from earlier phases — section headers,
plain-English explanations of Spark concepts, comments about what to look for
in the output.

**Explanation style:** Plain English. Non-technical analogies for systems concepts.
Excited about basketball — use the sport to motivate technical choices.

**Format:** When explaining multi-step procedures, numbered steps with code blocks.
When explaining concepts, prose. Don't over-bullet.

**Environment reminders to give Adam:**
- Always start cluster before submitting, stop immediately after
- Run from court-vision/ root directory
- Commit and sync to GitHub after each phase completes
- matplotlib.use('Agg') before importing pyplot; plt.savefig() not plt.show()
  (if any visualization is done on the cluster)
- Save notebooks to GCS not Local Disk

**AI attribution:** The course requires an attribution paragraph at the end of the
report stating what AI was used and how. Keep this in mind — do not produce
anything that would be academically dishonest beyond what's already disclosed.

---

## Environment Quirks — Things That Caused Problems

- First job after cluster start (within ~13 seconds) may run too fast/fail silently
  — wait a moment after the cluster shows RUNNING before submitting
- driveroutput streams to terminal in real-time but is also stored in GCS staging
  — if output doesn't appear inline, cat the driveroutput.000000000 file
- UCSB network may block downloads — use Google Cloud Shell as workaround
- PowerShell does not expand glob wildcards for external programs the same way bash does
  — gcloud handles the * itself; quote the path if in doubt
- cluster create takes ~90 seconds; cluster start (from stopped) takes ~60-90 seconds
- The cluster was previously deleted accidentally (confused stop with delete)
  — if `gcloud dataproc clusters list` is empty, recreate with:
  `gcloud dataproc clusters create mycluster --region=us-central1 --single-node --master-machine-type=e2-highmem-4 --optional-components=JUPYTER --enable-component-gateway`
  — if creation fails with permissions error, run the IAM fix first (see Technical Environment)
