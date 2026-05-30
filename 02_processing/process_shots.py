"""
process_shots.py
Phase 2 — Processing & Feature Engineering

Goal: take the raw CSVs from GCS, clean them, add engineered features,
and write the result as Parquet partitioned by season.

Every later phase (shot model, era analysis, archetypes) reads from
the Parquet output of this script — NOT from the slow raw CSVs.
Running this script once means every future job starts fast.

Two analytical outputs are also produced and saved to GCS:
  - Table 1: FG% and efficiency by zone         → report Section 4.1
  - Table 2: 3PT rate by season (era preview)   → report Section 4.2

Run from court-vision/ root in the VSCode terminal:
    gcloud dataproc clusters start mycluster
    gcloud dataproc jobs submit pyspark 02_processing/process_shots.py --cluster=mycluster
    gcloud dataproc clusters stop mycluster

Expected runtime: 5-10 minutes (the Parquet write step is the slow one)

Reads from : gs://pstat135-adam/raw/shots/NBA_*_Shots.csv
Writes to  : gs://pstat135-adam/processed/shots/         (Parquet, partitioned by season)
             gs://pstat135-adam/processed/tables/         (CSV summary tables)
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ── 1. Start the Spark session ────────────────────────────────────────────────
# Same pattern as verify_shots.py. On Dataproc, .getOrCreate() connects to
# the cluster's already-running Spark environment — memory, cores, and the
# GCS connector are pre-configured. You never set those manually.
spark = (
    SparkSession.builder
    .appName("court-vision: shot data processing")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# ── Path constants ─────────────────────────────────────────────────────────────
# Defining all GCS paths in one place at the top makes them easy to change
# and makes it obvious where data flows in and out of this script.
RAW_PATH       = "gs://pstat135-adam/raw/shots/NBA_*_Shots.csv"
PROCESSED_PATH = "gs://pstat135-adam/processed/shots/"
TABLES_PATH    = "gs://pstat135-adam/processed/tables/"

print(f"\n{'='*60}")
print("  Phase 2 — Shot Data Processing & Feature Engineering")
print(f"  Source  : {RAW_PATH}")
print(f"  Output  : {PROCESSED_PATH}")
print(f"{'='*60}\n")


# ── 2. Read the raw CSVs ──────────────────────────────────────────────────────
# The wildcard (*) loads all 21 season files into one combined DataFrame.
# This step is LAZY — Spark builds a plan for reading but does not actually
# read any data yet. Nothing happens until an action like .write() or .count()
# is called. This is one of Spark's core design principles: plan first,
# execute only when the result is needed.
print("Reading raw CSVs (lazy — no data loaded yet)...")
shots_raw = spark.read.csv(RAW_PATH, header=True, inferSchema=True)
print(f"  Columns recognized: {len(shots_raw.columns)}")
print(f"  Column names: {shots_raw.columns}\n")


# ── 3. Cast and clean ─────────────────────────────────────────────────────────
# .withColumn(new_name, expression) adds or replaces a column.
# We chain multiple .withColumn() calls — each one is still lazy.
# Spark combines all of them into a single optimized execution plan.
# No data has moved yet.
print("Defining casts and cleaning (still lazy)...")

shots_clean = (
    shots_raw

    # SHOT_MADE is boolean (true/false) in this dataset.
    # Spark MLlib requires numeric labels, so we cast to integer: true→1, false→0.
    # We create a new column SHOT_MADE_INT and keep the original SHOT_MADE
    # so the boolean version is still available for filtering and display.
    .withColumn("SHOT_MADE_INT", F.col("SHOT_MADE").cast("integer"))

    # GAME_DATE comes in as the string "MM-DD-YYYY" (e.g. "04-10-2022").
    # F.to_date() converts it to a proper date type using a format pattern:
    #   MM = two-digit month, dd = two-digit day, yyyy = four-digit year.
    # We keep SEASON_1 (integer) as the primary season identifier since it
    # already exists and is cleaner for groupBy. This parsed date is useful
    # for any game-level or monthly breakdowns later.
    .withColumn("GAME_DATE_PARSED", F.to_date(F.col("GAME_DATE"), "MM-dd-yyyy"))
)


# ── 4. Feature engineering ────────────────────────────────────────────────────
# We add computed columns that the models in Phases 3, 4, and 5 will use.
# All of these are derived from columns already in the data — none require
# external lookup tables or joins.
print("Defining feature engineering (still lazy)...")

shots_featured = (
    shots_clean

    # SECS_REMAINING: total seconds left in the current quarter.
    # Merges MINS_LEFT and SECS_LEFT into one clean continuous number.
    # Why combine: a model understands "90 seconds left" better than
    # two separate inputs of "1 minute" and "30 seconds."
    # Example: MINS_LEFT=2, SECS_LEFT=30 → SECS_REMAINING=150
    .withColumn(
        "SECS_REMAINING",
        F.col("MINS_LEFT") * 60 + F.col("SECS_LEFT")
    )

    # IS_3PT: 1 if this is a 3-point attempt, 0 if 2-point.
    # F.when(condition, value).otherwise(other_value) is PySpark's if-then-else.
    # This is the primary binary feature for both the era analysis and
    # the shot-make model — it is the column that captures the revolution.
    .withColumn(
        "IS_3PT",
        F.when(F.col("SHOT_TYPE") == "3PT Field Goal", 1).otherwise(0)
    )

    # SHOT_VALUE: how many points this attempt is worth if made (2 or 3).
    # This enables computing expected points per attempt by zone, which is
    # the exact arithmetic that proves mid-range shots are inefficient:
    #   Corner 3:   35% × 3 pts = 1.05 expected pts per attempt
    #   Mid-Range:  45% × 2 pts = 0.90 expected pts per attempt
    # The corner 3 is worth more despite a lower make percentage.
    .withColumn(
        "SHOT_VALUE",
        F.when(F.col("SHOT_TYPE") == "3PT Field Goal", 3).otherwise(2)
    )

    # IS_CLUTCH: 1 if this shot was taken in the final 2 minutes of
    # the 4th quarter or any overtime period, 0 otherwise.
    # Uses the & operator to combine two conditions (both must be true).
    # QUARTER >= 4 catches both 4th quarter (4) and overtime periods (5, 6...).
    # SECS_REMAINING <= 120 = 2 minutes (120 seconds) or less remaining.
    .withColumn(
        "IS_CLUTCH",
        F.when(
            (F.col("QUARTER") >= 4) & (F.col("SECS_REMAINING") <= 120), 1
        ).otherwise(0)
    )

    # EXPECTED_PTS: actual points scored on this specific attempt.
    # At the row level this is just SHOT_VALUE if made, 0 if missed.
    # When averaged across many shots (by zone, by player, by season),
    # it becomes the efficiency metric: average points per attempt.
    # This is the single most important number in the analytics movement.
    .withColumn(
        "EXPECTED_PTS",
        F.col("SHOT_VALUE") * F.col("SHOT_MADE_INT")
    )
)


# ── 5. Filter out backcourt shots ─────────────────────────────────────────────
# Backcourt heaves are half-court desperation throws at the buzzer.
# They are not real shot attempts — no player is trying to score from there,
# and including them would add noise to every efficiency metric and model.
# DECISION: filtering these out. Documented in DECISIONS.md.
#
# NOTE: .count() is an ACTION. This is the first point where Spark actually
# reads the raw CSV data and executes all the lazy transformations defined
# above. Everything from step 2 onward runs here for the first time.
print("\nFiltering backcourt shots (first action — data loads now)...")
backcourt_count = shots_featured.filter(F.col("BASIC_ZONE") == "Backcourt").count()
shots_processed = shots_featured.filter(F.col("BASIC_ZONE") != "Backcourt")
print(f"  Removed {backcourt_count:,} backcourt shots")
print(f"  Remaining shots: ~{4231262 - backcourt_count:,}\n")


# ── 6. FG% and efficiency by zone — Table 1 for the report ───────────────────
# The first real analytical output of the project.
# This table is the quantitative foundation of Act 1 of the story:
# "why did the mid-range shot become analytically undesirable?"
#
# .groupBy() splits the DataFrame into one group per unique BASIC_ZONE value.
# .agg() applies multiple aggregation functions to each group simultaneously:
#
#   F.count("*")         → total shot attempts from that zone
#   F.avg("SHOT_MADE_INT") → average of 0s and 1s = the make percentage
#   F.avg("EXPECTED_PTS")  → average points scored per attempt (efficiency)
#
# .orderBy(F.desc("pts_per_attempt")) sorts highest-efficiency zone first
# so the pattern (restricted area top, mid-range bottom) is immediately visible.
print("Computing Table 1: FG% and efficiency by zone...")

fg_by_zone = (
    shots_processed
    .groupBy("BASIC_ZONE")
    .agg(
        F.count("*").alias("attempts"),
        F.round(F.avg("SHOT_MADE_INT") * 100, 1).alias("fg_pct"),
        F.round(F.avg("EXPECTED_PTS"), 3).alias("pts_per_attempt"),
    )
    .orderBy(F.desc("pts_per_attempt"))
)

print("\nTABLE 1 — Efficiency by Zone (ordered best to worst):")
print("Look for: Restricted Area and Corner 3 at the top, Mid-Range near the bottom.\n")
fg_by_zone.show(truncate=False)

# Save to GCS as a CSV so it can be referenced without rerunning the job.
# coalesce(1) forces Spark to write a single file instead of many small ones.
# (By default, Spark splits output across many parallel files, one per core.
#  For a 7-row table that would create needless clutter.)
# The output is a folder — the actual CSV is the part-*.csv file inside it.
print("Saving Table 1 to GCS...")
(
    fg_by_zone
    .coalesce(1)
    .write
    .mode("overwrite")
    .option("header", "true")
    .csv(TABLES_PATH + "fg_pct_by_zone/")
)
print(f"  Saved to: {TABLES_PATH}fg_pct_by_zone/\n")


# ── 7. 3PT rate by season — era analysis preview ──────────────────────────────
# The centerpiece of Act 2: when did the revolution happen?
# This produces the numbers for Figure 2 in the report (the time-series plot).
# Phase 4 will visualize this; here we produce and save the underlying data.
#
# F.sum("IS_3PT") / F.count("*") = proportion of attempts that were 3-pointers.
# Multiplied by 100 → percentage.
# Sorted by SEASON_1 so the time trend reads chronologically.
print("Computing Table 2: 3PT rate by season (era analysis preview)...")

three_pt_by_season = (
    shots_processed
    .groupBy("SEASON_1")
    .agg(
        F.count("*").alias("total_attempts"),
        F.sum("IS_3PT").alias("three_pt_attempts"),
        F.round(F.sum("IS_3PT") / F.count("*") * 100, 1).alias("three_pt_rate_pct"),
        F.round(F.avg("SHOT_MADE_INT") * 100, 1).alias("overall_fg_pct"),
    )
    .orderBy("SEASON_1")
)

print("\nTABLE 2 — Three-Point Rate by Season (the revolution in numbers):")
print("Look for: a steady climb from ~2004, accelerating sharply around 2015-2016.\n")
three_pt_by_season.show(25, truncate=False)

print("Saving Table 2 to GCS...")
(
    three_pt_by_season
    .coalesce(1)
    .write
    .mode("overwrite")
    .option("header", "true")
    .csv(TABLES_PATH + "three_pt_rate_by_season/")
)
print(f"  Saved to: {TABLES_PATH}three_pt_rate_by_season/\n")


# ── 8. Write processed Parquet ────────────────────────────────────────────────
# This is the main output of Phase 2 — the cleaned, feature-rich dataset
# stored in Parquet format, ready for all future phases to read from.
#
# WHY PARQUET instead of CSV?
#   Columnar storage: when Phase 3 only needs LOC_X, LOC_Y, and SHOT_MADE_INT,
#   Spark reads only those three columns from disk — ignoring the other 30.
#   On a 4 million row dataset this makes queries dramatically faster.
#
#   Schema preserved: column types (integer, boolean, date) are saved inside
#   the file. No more inferSchema overhead on every read — types are already known.
#
#   Compressed: Parquet files are typically 5-10x smaller than equivalent CSVs.
#
#   Partition pruning: because we partitionBy("SEASON_1"), the data is split
#   into one subfolder per season. When Phase 4 filters to one season, Spark
#   reads only that folder and physically skips all other seasons on disk.
#
# .partitionBy("SEASON_1") creates this folder structure in GCS:
#   processed/shots/SEASON_1=2004/part-00000-....parquet
#   processed/shots/SEASON_1=2005/part-00000-....parquet
#   ...
#   processed/shots/SEASON_1=2024/part-00000-....parquet
#
# .mode("overwrite") replaces any existing files at that path.
# Safe to rerun — always produces a fresh, clean output.
print(f"Writing Parquet to {PROCESSED_PATH}")
print("(This is the slow step — typically 5-8 minutes. Output streams when done.)\n")

(
    shots_processed
    .write
    .mode("overwrite")
    .partitionBy("SEASON_1")
    .parquet(PROCESSED_PATH)
)

print("Parquet write complete.\n")


# ── 9. Verify the Parquet output ──────────────────────────────────────────────
# Read back the Parquet we just wrote and confirm it is correct.
# This proves the file is actually readable — not just that the write command
# ran without errors. When reading Parquet, Spark already knows the schema
# from the file itself, so no inferSchema is needed.
print("Verifying Parquet output...")
shots_verify = spark.read.parquet(PROCESSED_PATH)

verified_rows = shots_verify.count()
print(f"\nPARQUET VERIFICATION:")
print(f"  Rows written     : {verified_rows:,}")
print(f"  Columns          : {len(shots_verify.columns)}")
print(f"  Location         : {PROCESSED_PATH}")

print("\nProcessed schema (note SHOT_MADE_INT and new feature columns):")
shots_verify.printSchema()

print("\nRow count per season (confirms partition structure is correct):")
shots_verify.groupBy("SEASON_1").count().orderBy("SEASON_1").show(25)


# ── 10. Summary ───────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  PHASE 2 COMPLETE")
print(f"  Rows in Parquet  : {verified_rows:,}")
print(f"  Features added   : SHOT_MADE_INT, GAME_DATE_PARSED,")
print(f"                     SECS_REMAINING, IS_3PT, SHOT_VALUE,")
print(f"                     IS_CLUTCH, EXPECTED_PTS")
print(f"  Tables saved     : fg_pct_by_zone, three_pt_rate_by_season")
print(f"  Parquet path     : {PROCESSED_PATH}")
print(f"")
print(f"  All future phases read from the Parquet path above.")
print(f"  The raw CSVs are no longer needed for analysis.")
print(f"{'='*60}\n")

spark.stop()
