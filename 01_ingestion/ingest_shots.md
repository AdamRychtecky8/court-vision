# Shot Data Ingestion Record
## court-vision | Phase 1

*This file is the reproducibility record for Phase 1.
Anyone should be able to follow these steps to recreate the raw data in GCS from scratch.*

---

## Data Sources

| Dataset | Source | URL |
|---------|--------|-----|
| NBA Shot Locations (2004–2024) | DomSamangy / GitHub | https://github.com/DomSamangy/NBA_Shots_04_25 |
| Kaggle mirror | mexwell / Kaggle | https://www.kaggle.com/datasets/mexwell/nba-shots |
| Original data | NBA Stats API | https://stats.nba.com |

**Who collected it:** Vladislav Shufinskiy and Dom Samangy compiled shot-chart detail
from the NBA's public stats API, covering every field goal attempt tracked
since the 2003-04 season.

---

## Download

Downloaded manually from Kaggle (mexwell/nba-shots).
Files are delivered as individual CSVs per season, named `NBA_YYYY_Shots.csv`.

**Note on UCSB network:** If downloading on campus, the UCSB network may block
outbound downloads from Kaggle or GitHub. Workaround: use Google Cloud Shell
(the browser terminal at console.cloud.google.com) to download and pipe
directly to GCS, bypassing the campus restriction.

---

## Upload to GCS

All 21 CSV files uploaded from local `data/raw/` to the raw shots bucket path.
The local `data/` folder is gitignored — raw data is never committed to the repo.

**Upload command (run from court-vision/ root):**
```powershell
gcloud storage cp data\raw\NBA_*_Shots.csv gs://pstat135-adam/raw/shots/
```

**Verify files landed:**
```powershell
gcloud storage ls gs://pstat135-adam/raw/shots/
```

Files confirmed in GCS:
```
gs://pstat135-adam/raw/shots/NBA_2004_Shots.csv
gs://pstat135-adam/raw/shots/NBA_2005_Shots.csv
...
gs://pstat135-adam/raw/shots/NBA_2024_Shots.csv
```
(21 files total, 2004–2024)

---

## Verification Results

Verification script: `01_ingestion/verify_shots.py`
Run date: 2026-05-30
Cluster: mycluster (single-node e2-highmem-4, us-central1)

| Metric | Result |
|--------|--------|
| Total rows | 4,231,262 |
| Total columns | 26 |
| Seasons covered | 2004–2024 (all 21) |
| Nulls in LOC_X | 0 |
| Nulls in LOC_Y | 0 |
| Nulls in SHOT_TYPE | 0 |
| Nulls in GAME_DATE | 0 |
| 2PT Field Goal attempts | 3,025,063 (71.5%) |
| 3PT Field Goal attempts | 1,206,199 (28.5%) |

**Season row counts:**

| Season | Rows | Notes |
|--------|------|-------|
| 2004 | 189,803 | |
| 2005 | 197,626 | |
| 2006 | 194,314 | |
| 2007 | 196,072 | |
| 2008 | 200,501 | |
| 2009 | 199,030 | |
| 2010 | 200,966 | |
| 2011 | 199,761 | |
| 2012 | 161,205 | Shortened season — NBA lockout (66 games) |
| 2013 | 201,579 | |
| 2014 | 204,126 | |
| 2015 | 205,550 | |
| 2016 | 207,893 | |
| 2017 | 209,929 | |
| 2018 | 211,707 | |
| 2019 | 219,458 | |
| 2020 | 188,116 | COVID bubble season (fewer games) |
| 2021 | 190,983 | |
| 2022 | 216,722 | |
| 2023 | 217,220 | |
| 2024 | 218,701 | |

---

## Key Discoveries (document in DECISIONS.md)

Column names differ from the NBA Stats API standard. All Phase 2+ code
must use the actual names confirmed here:

| Expected (NBA API standard) | Actual column name | Note |
|-----------------------------|--------------------|------|
| `SHOT_MADE_FLAG` | `SHOT_MADE` | **boolean, not integer — cast in Phase 2** |
| `SHOT_ZONE_BASIC` | `BASIC_ZONE` | string |
| `SHOT_ZONE_AREA` | `ZONE_NAME` | string |
| `SHOT_ZONE_RANGE` | `ZONE_RANGE` | string |

`SEASON_1` (integer year) and `SEASON_2` (formatted string e.g. "2021-22")
both exist — use `SEASON_1` for all groupBy and sorting operations.

---

## GCS Location

```
gs://pstat135-adam/raw/shots/NBA_*_Shots.csv   ← raw input (this phase)
gs://pstat135-adam/processed/shots/            ← Parquet output (Phase 2)
```
