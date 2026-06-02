# Master Guide — PSTAT 135 Final Project
### NBA Play-by-Play Analytics on Google Cloud
*Adam Rychtecky — living reference document*

---

## How to use this document

This is the single source of truth for the final project. It holds (1) the story I'm telling, (2) a phase-by-phase plan from nothing to a finished, professionally-presentable repo, and (3) a context block any AI assistant can read to help me effectively. Update it as decisions are made. If an instruction here ever conflicts with the official Final Project page or Prof. Ichiba's rubric, the official source wins — flag the conflict and update this file.

---

## Part 1 — The Story

### The core question (lead with this everywhere)
**How did the three-point revolution reshape the NBA — why it made sense, when it happened, and what it did to the kinds of players who succeed?**

### The three-act narrative

For most of NBA history the mid-range jump shot was a core weapon. Then in the 2010s the league largely abandoned it: shoot at the rim or shoot threes, avoid everything in between. This project is the quantitative story of how and why that happened, in three connected acts.

**Act 1 — Why it happened (shot value).** A shot's worth is its make-probability times its point value. A three is worth 1.5× a two, and the highest-percentage shots are at the rim, so the mid-range is the worst of both worlds — not close enough to be high-percentage, not far enough to earn the extra point. The shot-make model (LASSO feature selection → classifier) tests this directly. *Expected finding:* location dominates context; restricted-area and corner-3 features carry the most weight; mid-range looks weak. *Honest caveat:* no defender data, so this measures shot **location** value, not full shot **quality**.

**Act 2 — When it happened (timing).** If the math was always there, why now? Because the Warriors proved a style built around it wins. The era analysis plots three-point attempt rate by season and finds the inflection point. *Expected finding:* a clear upward bend in the mid-2010s aligning with Curry's MVP seasons and the 2015–2016 title run. This is a finding I can *explain*, not just plot.

**Act 3 — What it did (player reorganization).** When the optimal shot changed, the optimal player changed. PCA + K-Means clusters players by statistical profile; clustering pre-2015 and post-2015 separately reveals the shift. *Expected finding:* the "stretch big" archetype appears post-2015; the pure mid-range specialist and back-to-the-basket center shrink.

**Through-line in one sentence:** *the NBA realized certain shots were mathematically more valuable, the Warriors proved building around them wins, the league copied it, and the kind of player who thrives changed as a result.* The three analyses are the evidence for each link.

**Honesty note (graded favorably):** these are hypotheses. The break may be earlier or fuzzier than expected; the trend predates the Warriors, who accelerated more than started it. Clusters may not split as cleanly as the labels suggest. Document what is actually found, including where the data complicates the clean story.

---

## Part 2 — Guiding Principles (from Lecture 18 + the writing standard)

These are the explicit criteria the work will be judged against. Pin them to the wall.

1. **Tell a coherent, curated story.** Every figure and table serves the core question. Cut anything that doesn't.
2. **Lead with the core question.** State it in the abstract, the intro, the README, and the video's first 20 seconds.
3. **Present information hierarchically.** Headline finding first, then evidence, then detail. Descriptive stats before models.
4. **Use literate statistical programming.** Code, output, and narrative live together (Jupyter notebooks; the report references them). Not code dumps with no explanation.
5. **Ensure full reproducibility.** Anyone can rerun from documented commands: pinned versions, fixed random seeds, recorded data-fetch steps, a clear run order.

Plus the report-writing standard (from `statistical-writing.md`): informative title; ~150-word abstract written last; general→specific intro ending in goals; methods detailed enough to replicate; results as journal-style prose with labeled, referenced figures and **no raw console output**; honest limitations; APA references. Use Gelman's order: conclusions → supporting figures → methods → intro → abstract.

### The instructor's demonstrated lifecycle (mirror it)
Lecture 18's examples all follow: **acquire data → convert to Parquet → profile & clean → unsupervised (clustering) + supervised (regression/classification) → evaluate with standard metrics (RMSE / AUC / F1 / confusion matrix) → visualize (PCA scatter, predicted-vs-actual, confusion matrix, maps) → summarize insights.** This project follows the same arc, which is the safest path to full marks because it matches what the instructor demonstrated as "good."

---

## Part 3 — The Data

| Dataset | Source | Coverage | Feeds which analysis |
|---|---|---|---|
| Shot locations | github.com/DomSamangy/NBA_Shots_04_25 | 2003/04–2024/25, ~4M shots, LOC_X/Y | Shot-make model + era analysis |
| Play-by-play | github.com/shufinskiy/nba_data | 1996/97–2023/24, ~14M events | Player archetypes + (optional) assist network |
| Kaggle mirror | kaggle.com/datasets/mexwell/nba-shots | — | Fallback download |
| Original source | stats.nba.com | — | Provenance / citation |

**Key fields.** Shots: `LOC_X`, `LOC_Y`, `SHOT_DISTANCE`, `SHOT_ZONE_BASIC`, `SHOT_TYPE`, `ACTION_TYPE`, `SHOT_MADE_FLAG`, `MINS_LEFT`, `QUARTER`. Play-by-play: `GAME_ID`, `EVENTMSGTYPE` (1=made, 2=missed, 4=rebound, 5=turnover…), `PERIOD`, `PLAYER1/2/3_ID`, `SCOREMARGIN`. **Note:** `SEASON_YEAR` likely must be derived from `GAME_DATE` or filename — verify before any season-level groupBy.

**Why distributed processing is warranted (state honestly):** a single season fits on a laptop; the Spark justification is full-28-season PCA/K-Means across all player-seasons and aggregations over all ~18M rows.

**Known limitations to document:** no defender/contest tracking; known play-by-play attribution errors (rebounds/assists); cross-era rule changes (2004/05 hand-check) confound comparisons; join on `PLAYER_ID`, never name strings.

---

## Part 4 — Scoped Analysis Plan

Do three things well rather than six things shallowly. The report's results section is 5–8 pages; depth beats breadth.

| Analysis | Method | Course lecture | Output |
|---|---|---|---|
| Shot-make prediction | LASSO (ElasticNet, α=1) → LogisticRegression + LinearSVC | 7, 8 | Feature ranking, ROC-AUC, F1, confusion matrix |
| Player archetypes | Standardize → PCA → K-Means (elbow/silhouette) | 9, 10, 12 | Scree plot, PC1–PC2 scatter, labeled clusters |
| Era analysis | Time-series aggregation of 3PT/mid/rim rates by season | feature eng. + viz | Inflection-point plot, zone heatmaps |
| Robustness/fairness | Disaggregate model metrics by era & zone | 16 | Comparison table |
| **Optional** assist network | GraphFrames PageRank | 13 | Top playmakers — future-work/appendix only |

GraphFrames is the strongest portfolio piece but the riskiest to install on Dataproc (must match the `graphframes` package to the Spark version via `--packages`). Keep it as future-work unless the core finishes early.

---

## Part 5 — Execution Plan: from nothing to done

Each phase has a goal, concrete steps, and a "done when." Build a **thin slice first** (one result end-to-end) before widening.

### Phase 0 — Repo + environment scaffolding (Day 0)
- `git init` locally; create the folder structure (Part 6); add `.gitignore` and a stub `README.md` and `DECISIONS.md` before any analysis.
- Create the GitHub repo (private to start; flip public when presentable). Do **not** commit data or credentials.
- Confirm GCP is live: project `pstat135-hw-497220`, bucket `pstat135-adam`, cluster `mycluster`.
- **Done when:** empty-but-structured repo pushed; cluster can start and run a trivial PySpark job.

### Phase 1 — Ingestion to Cloud Storage (Days 1)
- Download datasets and upload to GCS. **UCSB network may block outbound downloads** (confirmed previously with California Housing) — mitigate by downloading inside **Cloud Shell** (GCP-side, bypasses campus network) straight to GCS, or use the Kaggle mirror, or download off-campus.
  ```bash
  gsutil cp nba_shots_2004_2025.csv gs://pstat135-adam/raw/shots/
  gsutil cp -r pbp_seasons/ gs://pstat135-adam/raw/pbp/
  ```
- Record exact fetch commands in `01_ingestion/` for reproducibility.
- **Done when:** raw CSVs are in `gs://pstat135-adam/raw/` and row counts are logged.

### Phase 2 — Processing & feature engineering (Days 2–4)
- PySpark on Dataproc: load CSVs, clean nulls, validate `EVENTMSGTYPE` distributions, derive `SEASON_YEAR`, engineer shot distance / zone flags / context features, aggregate play-by-play to player-season per-36 profiles.
- Write processed output as **Parquet partitioned by season** to `gs://pstat135-adam/processed/`.
- *(Recommended, instructor-aligned)* load processed tables into **BigQuery** so EDA can be SQL and it mirrors Lecture 18's examples.
- **Done when:** `SELECT SHOT_ZONE_BASIC, AVG(SHOT_MADE_FLAG) ... GROUP BY` returns sensible FG% by zone.

### Phase 3 — Descriptive + era analysis = the thin slice (Days 5–6)
- Table of FG% and frequency by zone; shot volume by season; the 3PT-rate-by-season plot (Figure 2 / the centerpiece).
- **Done when:** the era plot renders and the inflection point is identified. *This is the first real, writable result.*

### Phase 4 — Supervised: shot-make model (Days 7–9)
- LASSO for feature selection, then LogisticRegression + LinearSVC. Train/test split, fixed seed. Evaluate ROC-AUC, F1, confusion matrix.
- **Done when:** feature ranking and classifier metrics are produced and interpretable.

### Phase 5 — Unsupervised: archetypes (Days 10–12)
- Standardize player-season features → PCA (report variance explained) → K-Means (choose k by elbow/silhouette) → label clusters. Cluster pre-/post-2015 separately and compare.
- **Done when:** scree plot, PC scatter, and labeled clusters exist; the era comparison is interpretable.

### Phase 6 — Robustness / fairness (Day 13)
- Disaggregate shot-model metrics by era and zone.
- **Done when:** comparison table produced; any uneven performance noted.

### Phase 7 — Report (Days 14–17)
- Fill the report skeleton in Gelman's order. Pull figures from notebooks. No console output in the body.
- Word / RMarkdown / LaTeX → PDF. Include the required AI-use attribution paragraph.
- **Done when:** ~10-page report reads as one coherent story and passes the "friend test."

### Phase 8 — GauchoCast video (Days 18–19)
- Short presentation: core question first, then the three acts, then findings and limitations. Record and upload to GauchoCast.
- **Done when:** video uploaded; report submitted to Gradescope.

### Phase 9 — Portfolio polish (Days 20–21)
- README findings-first; finish `DECISIONS.md`; clean module structure; verify a stranger can run it from the README; flip repo public.
- **Done when:** repo URL can be handed to someone who's never heard of it and they understand it without asking a question.

Keep stopping the cluster after every session.

---

## Part 6 — Git Repo Structure (the professional setup)

```
nba-bigdata-pipeline/
├── README.md                  # findings-first; the front door
├── DECISIONS.md               # one paragraph per non-obvious choice (interview prep)
├── .gitignore                 # excludes data, credentials, checkpoints
├── requirements.txt           # pinned versions for reproducibility
├── 01_ingestion/              # download + gsutil upload scripts (commands recorded)
├── 02_processing/             # PySpark load, clean, feature engineering
├── 03_shot_model/             # LASSO + classification
├── 04_era_analysis/           # 3-point revolution time series + heatmaps
├── 05_clustering/             # PCA + K-Means archetypes
├── 06_networks/               # (optional) GraphFrames assist network
├── 07_evaluation/             # fairness/robustness audit
├── notebooks/                 # literate analysis notebooks (mirrored from GCS)
├── data/
│   └── README.md              # schema of every processed column (NO data files)
├── results/
│   ├── figures/               # exported PNGs used in the report
│   └── key_findings.md        # 3–5 plain-English findings
└── report/                    # final PDF + source (LaTeX/Rmd/docx)
```

**`.gitignore` must include:** `data/*.csv`, `data/*.parquet`, `*.json` (credentials), `.ipynb_checkpoints/`, `__pycache__/`, `venv/`, `*.crc`. GitHub rejects files over 100 MB anyway — the ~2–3 GB datasets stay in GCS; the repo documents how to fetch them.

**README layout (findings-first):** one-line pitch → core question → 3–5 key findings *with numbers* → architecture diagram → how to run → repo structure → tech stack → limitations → AI-use note.

**`DECISIONS.md`** is the highest-value habit: why Parquet, why partition by season, why LASSO over plain logistic, why k was chosen as it was, how the `SEASON_YEAR` derivation was handled. This file *is* the interview.

---

## Part 7 — Tech Stack & Storage Map

| Stage | Technology | Lives where |
|---|---|---|
| Raw storage | Google Cloud Storage | `gs://pstat135-adam/raw/` |
| Compute | Dataproc (managed Spark) | `mycluster`, single-node `e2-highmem-4`, `us-central1` |
| Processing | PySpark (Spark SQL + DataFrames) | On Dataproc |
| ML | Spark MLlib (LASSO, LogReg, LinearSVC, PCA, K-Means) | On Dataproc |
| Graph (optional) | GraphFrames | On Dataproc |
| Processed storage | Parquet, partitioned by season | `gs://pstat135-adam/processed/` |
| Query/EDA layer | BigQuery (instructor-aligned) | project `pstat135-hw-497220` |
| Notebooks | JupyterLab on Dataproc | **save to GCS, not Local Disk** |
| Visualization | matplotlib (`Agg` backend) + seaborn | `plt.savefig()` |
| Version control | Git + GitHub | public repo |
| Report | Word / RMarkdown / LaTeX → PDF | Gradescope |
| Presentation | GauchoCast | GauchoCast |

**Submission vs. portfolio:** report PDF → Gradescope (and `report/` in repo); video → GauchoCast (link in README); code → Appendix excerpts (and full in repo); raw/processed data → GCS only, never Git.

---

## Part 8 — Environment Quirks (hard-won; don't relearn these)

- `matplotlib.use('Agg')` **before** importing pyplot; save with `plt.savefig()` — inline display won't render on the cluster.
- Save notebooks to **GCS**, not Local Disk — Dataproc local disk is ephemeral.
- **Stop the cluster manually** after every session; don't rely on auto-idle.
- UCSB network may block outbound dataset downloads — use Cloud Shell or mirrors.
- Join player records on `PLAYER_ID`, never on name strings.

---

## Part 9 — Context for an AI assistant

*Paste or point any AI helper here before asking for help.*

**Who/what:** Adam, UCSB student in PSTAT 135 (Big Data Analytics, Prof. Ichiba). Not a CS major — prefers systems concepts explained in plain terms. Final project = a written report (~10 pages, Word/Rmd/LaTeX) + a short GauchoCast video, due finals week, submitted to Gradescope + GauchoCast. The grade is on the report and presentation, not on a codebase.

**Working style:** walkthrough style — explain concepts alongside code, not finished notebooks handed over. Efficient and direct; structured breakdowns without filler.

**The learning rules (respect these — they are the point of the project):**
1. *Write before you run.* Adam writes the first attempt at any significant analytical code himself. The AI helps him understand what's wrong — it does not rewrite it for him.
2. *The three questions* before any AI-suggested code is used: What does it do, step by step? What would break it? How would I know if it was silently producing wrong results?
3. *Memory test:* after each phase Adam describes the whole system from memory.
4. *DECISIONS.md as you go.*

**AI-use policy (course rule):** limited, attributed AI use only — brainstorming, concept explanation, source-finding, tutoring, summarizing, editing, feedback. A paragraph at the end of the work must state what tool was used and how. So: an AI agent should help with **planning, explanation, debugging code Adam wrote, and review** — and should **not** produce the core analytical code for him. Infrastructure commands (gsutil, git, cluster start/stop) are fine to provide fully; the LASSO/PCA/K-Means logic he writes himself.

**Scope guardrail:** three core analyses (shot-make, archetypes, era) + a short fairness check. Don't expand scope; depth over breadth. GraphFrames is optional future-work.

**The five judging principles:** coherent curated story; lead with the core question; hierarchical presentation; literate statistical programming; full reproducibility.

**Environment facts:** GCP project `pstat135-hw-497220`; bucket `pstat135-adam`; cluster `mycluster` (single-node `e2-highmem-4`, `us-central1`); stack is Dataproc + PySpark + MLlib + GCS Parquet + (optionally) BigQuery + JupyterLab + matplotlib. Honor the environment quirks in Part 8. Prep homework is done (penalized regression, classification, dimensionality reduction on the Wine dataset).

**Data facts:** see Part 3 — which dataset feeds which analysis, the `SEASON_YEAR` derivation gotcha, and the documented limitations (no defender data, play-by-play errors, era confounds).

---

## Part 10 — Reproducibility Checklist (Lecture 18 principle #5)

- [ ] `requirements.txt` with pinned versions committed.
- [ ] All random operations use a fixed seed.
- [ ] Exact data-fetch commands recorded in `01_ingestion/`.
- [ ] `data/README.md` documents every processed column.
- [ ] README states the run order start to finish.
- [ ] A clean checkout + the README is enough for a stranger to reproduce the results.

---

## Update Log
| Date | Change |
|---|---|
| May 2026 | Master guide created; aligned to Lecture 18 lifecycle and official report format. |
| *next* | *update when figures/findings land and when the official rubric is confirmed on Canvas.* |
