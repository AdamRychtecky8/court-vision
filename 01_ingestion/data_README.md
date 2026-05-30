# Data Dictionary — NBA Shot Data
## court-vision | data/README.md

*Authoritative column reference for all phases and for the report's Data section.
Schema confirmed from verify_shots.py output on 2026-05-30.*

---

## Dataset Overview

| Property | Value |
|----------|-------|
| Source | DomSamangy/NBA_Shots_04_25 (via Kaggle mirror mexwell/nba-shots) |
| GCS raw path | `gs://pstat135-adam/raw/shots/NBA_*_Shots.csv` |
| GCS processed path | `gs://pstat135-adam/processed/shots/` (Parquet, written in Phase 2) |
| Coverage | 2003-04 through 2024-25 NBA seasons |
| Total rows | 4,231,262 |
| Total columns | 26 |
| Unit of observation | One row = one field goal attempt |

---

## Column Reference

| Column | Type | Description | Phase 2+ Notes |
|--------|------|-------------|----------------|
| `SEASON_1` | integer | Season start year (e.g. 2022 for 2021-22). **Primary season identifier** | Use for all groupBy and sorting |
| `SEASON_2` | string | Formatted season string (e.g. "2021-22") | Human-readable label only |
| `TEAM_ID` | integer | NBA franchise identifier | Join key |
| `TEAM_NAME` | string | Full team name (e.g. "Los Angeles Lakers") | |
| `PLAYER_ID` | integer | NBA player identifier | Join key for player-season grouping |
| `PLAYER_NAME` | string | Player's full name | Group by `PLAYER_NAME + SEASON_1` for archetypes |
| `POSITION_GROUP` | string | Broad position group: G (guard), F (forward), C (center) | Useful for labeling archetypes |
| `POSITION` | string | Specific position (e.g. "SG-PG", "PF") | |
| `GAME_DATE` | string | Date in MM-DD-YYYY format (e.g. "04-10-2022") | Parse with `to_date` if needed; use `SEASON_1` instead where possible |
| `GAME_ID` | integer | Unique game identifier | |
| `HOME_TEAM` | string | Home team abbreviation (e.g. "DEN") | |
| `AWAY_TEAM` | string | Away team abbreviation (e.g. "LAL") | |
| `EVENT_TYPE` | string | "Made Shot" or "Missed Shot" | Redundant with `SHOT_MADE` |
| `SHOT_MADE` | boolean | true = made, false = missed | **Cast to integer for MLlib:** `F.col("SHOT_MADE").cast("integer")` |
| `ACTION_TYPE` | string | Specific shot action (e.g. "Jump Shot", "Layup Shot", "Running Dunk Shot") | One-hot encode for shot model |
| `SHOT_TYPE` | string | "2PT Field Goal" or "3PT Field Goal" | Primary era-analysis feature |
| `BASIC_ZONE` | string | Court zone (see values below). **Use this, not `SHOT_ZONE_BASIC`** | Primary zone feature |
| `ZONE_NAME` | string | Specific sub-zone within BASIC_ZONE (e.g. "Center", "Left Side Center") | |
| `ZONE_ABB` | string | Zone abbreviation (e.g. "C", "LC", "RC", "L", "R") | |
| `ZONE_RANGE` | string | Distance band (e.g. "Less Than 8 ft.", "8-16 ft.", "16-24 ft.", "24+ ft.") | |
| `LOC_X` | double | Horizontal court coordinate from basket | Non-null; use with LOC_Y for shot location plots |
| `LOC_Y` | double | Vertical court coordinate from basket | Non-null; scale confirmed during Phase 3 visualization |
| `SHOT_DISTANCE` | integer | Shot distance in feet | Continuous feature for shot model |
| `QUARTER` | integer | Game quarter (1–4; 5+ = overtime) | Context feature; can flag clutch time |
| `MINS_LEFT` | integer | Minutes remaining in current quarter | Context feature |
| `SECS_LEFT` | integer | Seconds remaining in current minute | Combine with MINS_LEFT for total seconds |

---

## BASIC_ZONE Values

These are the seven zone categories. All appear with non-zero counts:

| Zone | Type | Notes |
|------|------|-------|
| Restricted Area | 2PT | Directly at the basket; highest FG% |
| In The Paint (Non-RA) | 2PT | Paint shots outside the restricted area |
| Mid-Range | 2PT | The "worst shot" in analytics terms |
| Left Corner 3 | 3PT | High efficiency 3PT zone |
| Right Corner 3 | 3PT | High efficiency 3PT zone |
| Above the Break 3 | 3PT | Highest volume 3PT zone |
| Backcourt | 3PT | Half-court heaves; usually filtered out |

---

## Key Notes for Phase 2

**SHOT_MADE is boolean — must cast before MLlib:**
```python
from pyspark.sql import functions as F
shots = shots.withColumn("SHOT_MADE_INT", F.col("SHOT_MADE").cast("integer"))
```

**Total game-clock seconds remaining:**
```python
shots = shots.withColumn(
    "SECS_REMAINING",
    F.col("MINS_LEFT") * 60 + F.col("SECS_LEFT")
)
```

**Player-season grouping key for archetypes (Phase 5):**
```python
shots.groupBy("PLAYER_NAME", "PLAYER_ID", "SEASON_1", "POSITION_GROUP")
```

**Filter out backcourt shots for the model (optional):**
```python
shots = shots.filter(F.col("BASIC_ZONE") != "Backcourt")
```

---

## Known Limitations

- No defender/contest data — model captures shot **location** value, not true shot **quality**
- Play-by-play assists and rebounds are NOT in this dataset (would require separate ingestion of shufinskiy/nba_data for the optional GraphFrames assist network)
- 2012 season shortened to 66 games (NBA lockout) — row count of 161,205 is expected, not missing data
- 2020 season shortened due to COVID bubble
- Cross-era rule changes (2004-05 hand-check rules) complicate direct season comparisons — document in report limitations

---

## What Is NOT in This Dataset

For context, these analyses would require additional data:

| Analysis | Missing data | Source if needed |
|----------|-------------|-----------------|
| Assist network (GraphFrames) | Play-by-play events | shufinskiy/nba_data |
| Rebounds, blocks, steals | Box score stats | basketball-reference |
| Defender at time of shot | SportVU tracking | Not publicly available |
