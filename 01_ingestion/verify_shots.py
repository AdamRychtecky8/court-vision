"""
verify_shots.py
Phase 1 — Ingestion Verification

Goal: confirm the raw shot CSVs loaded from GCS are structurally sound
before any processing begins. This is NOT analysis — it is a sanity check.
Run this after uploading all NBA_YYYY_Shots.csv files to GCS.

Run from the court-vision/ root in the VSCode terminal:
    gcloud dataproc clusters start mycluster
    gcloud dataproc jobs submit pyspark 01_ingestion/verify_shots.py --cluster=mycluster
    gcloud dataproc clusters stop mycluster

Expected output:
    - ~4 million total rows across 2004-2024
    - Columns including LOC_X, LOC_Y, SHOT_MADE_FLAG, SHOT_ZONE_BASIC
    - Restricted area should have the highest FG% in the zone breakdown
    - Zero or near-zero nulls in SHOT_MADE_FLAG (the target variable)
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ── 1. Start a Spark session ──────────────────────────────────────────────────
# SparkSession is the single entry point to everything in PySpark.
# On Dataproc, .getOrCreate() picks up the cluster's existing configuration
# automatically — memory, cores, and the GCS connector are already set up.
# You never configure those manually when running on a managed cluster.
spark = (
    SparkSession.builder
    .appName("court-vision: shot data verification")
    .getOrCreate()
)

# Reduces log noise so the output you care about is easy to read.
# INFO level prints hundreds of internal Spark lines; WARN only shows problems.
spark.sparkContext.setLogLevel("WARN")


# ── 2. Read all 21 season CSVs in a single call ───────────────────────────────
# The wildcard (*) tells Spark to match every file that fits the pattern.
# All 21 files load into one DataFrame — Spark treats them as a single table
# even though they are stored as separate files on GCS.
#
# header=True    → the first row of each CSV contains column names, not data
# inferSchema=True → Spark reads a sample of each column to guess its type
#                   (integer, double, string, etc.) instead of defaulting
#                   everything to string. This takes a bit longer but gives
#                   you properly-typed columns from the start.
RAW_PATH = "gs://pstat135-adam/raw/shots/NBA_*_Shots.csv"

print(f"\n{'='*60}")
print(f"  Loading: {RAW_PATH}")
print(f"{'='*60}\n")

shots = spark.read.csv(RAW_PATH, header=True, inferSchema=True)


# ── 3. Schema ─────────────────────────────────────────────────────────────────
# printSchema() shows every column name and its inferred data type.
# Key things to verify when you read this output:
#   SHOT_MADE_FLAG  → should be IntegerType (0 or 1), not StringType
#   LOC_X, LOC_Y   → should be a numeric type (the court coordinates in tenths of a foot)
#   GAME_DATE       → string or date — note which, you will need to parse it in Phase 2
#   SEASON or YEAR  → check whether a ready-made season column exists
#                     (look for anything with "SEASON" or "YEAR" in the name)
print("SCHEMA:")
shots.printSchema()


# ── 4. Row count ─────────────────────────────────────────────────────────────
# .count() is a Spark "action" — it triggers actual distributed computation.
# Everything written before this line was lazy: Spark built a plan for the work
# but did not execute it yet. The moment you call an action like count(), show(),
# or write(), Spark executes the whole plan across the cluster.
# Expected: somewhere around 4 million rows for 2004-2024.
total_rows = shots.count()
print(f"\nTOTAL ROWS: {total_rows:,}")


# ── 5. Sample rows ───────────────────────────────────────────────────────────
# .show() prints rows in a formatted table so you can visually inspect the data.
# truncate=False shows the full value of each cell instead of cutting it off.
# This is the quickest check that the data looks like real basketball records
# and not shifted columns, garbled characters, or header rows mixed in as data.
print("\nSAMPLE ROWS (first 5):")
shots.show(5, truncate=False)


# ── 6. Null check on the columns that matter most ────────────────────────────
# Nulls in SHOT_MADE_FLAG are the most dangerous: it is the target variable
# for the shot-make model. Any null there is a row we cannot train or evaluate on.
# Nulls in LOC_X / LOC_Y mean we cannot plot or zone-classify that shot.
#
# F.col(c).isNull() produces a boolean column (True where null, False elsewhere).
# .cast("int") converts True→1, False→0, so summing it gives a null count.
# The guard "if c in shots.columns" skips a column if it does not exist under
# that exact name — protects against the script crashing on a name mismatch.
key_cols = [
    "SHOT_MADE_FLAG",
    "LOC_X",
    "LOC_Y",
    "SHOT_ZONE_BASIC",
    "SHOT_TYPE",
    "GAME_DATE",
]

print("\nNULL COUNTS IN KEY COLUMNS:")
null_counts = shots.select([
    F.sum(F.col(c).isNull().cast("int")).alias(c)
    for c in key_cols
    if c in shots.columns
])
null_counts.show()


# ── 7. Shot zone sanity check ─────────────────────────────────────────────────
# This is not analysis — it is a plausibility check.
# FG% by zone should follow the pattern we know from basketball:
#   Restricted Area  ~60-65%  (highest)
#   Corner 3         ~38-40%
#   Mid-Range        ~40-44%  (lowest efficiency per point)
# If these numbers look wildly off, something is wrong with the data.
#
# F.avg("SHOT_MADE_FLAG") works because SHOT_MADE_FLAG is 0 or 1:
# the average of a binary column is the proportion of 1s, i.e. the make rate.
# Multiplying by 100 converts to a percentage.
if "SHOT_ZONE_BASIC" in shots.columns:
    print("\nSHOT ATTEMPTS AND FG% BY ZONE (sanity check):")
    (
        shots
        .groupBy("SHOT_ZONE_BASIC")
        .agg(
            F.count("*").alias("attempts"),
            F.round(F.avg("SHOT_MADE_FLAG") * 100, 1).alias("fg_pct"),
        )
        .orderBy(F.desc("attempts"))
        .show()
    )


# ── 8. Three-point vs two-point split ────────────────────────────────────────
# Quick check that shot type is sensible and parseable.
# Expect roughly 35-40% of shots to be 3-pointers in recent seasons,
# lower in earlier seasons — this will be central to the era analysis in Phase 3.
if "SHOT_TYPE" in shots.columns:
    print("\nSHOT TYPE BREAKDOWN:")
    shots.groupBy("SHOT_TYPE").count().orderBy(F.desc("count")).show()


# ── 9. Season coverage check ─────────────────────────────────────────────────
# Confirm all 21 seasons are present and the row counts look plausible.
# If a season is missing entirely, the era analysis will have a gap.
#
# The script auto-detects whether a season column already exists (look for
# any column with "SEASON" or "YEAR" in the name). If one exists, group by it.
# If not, sample GAME_DATE so you know what format to parse in Phase 2.
season_candidates = [
    c for c in shots.columns
    if "SEASON" in c.upper() or "YEAR" in c.upper()
]

if season_candidates:
    season_col = season_candidates[0]
    print(f"\nSEASON COLUMN FOUND: '{season_col}'")
    print("ROW COUNT PER SEASON (should be 21 seasons, 2004-2024):")
    (
        shots
        .groupBy(season_col)
        .count()
        .orderBy(season_col)
        .show(30)
    )
else:
    print("\nNO SEASON COLUMN FOUND.")
    print("Note for Phase 2: derive season from GAME_DATE.")
    print("GAME_DATE sample values:")
    shots.select("GAME_DATE").distinct().orderBy("GAME_DATE").show(10)


# ── 10. Summary ───────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  VERIFICATION COMPLETE")
print(f"  Total rows : {total_rows:,}")
print(f"  Columns    : {len(shots.columns)}")
print(f"  Source     : {RAW_PATH}")
print(f"{'='*60}\n")

spark.stop()