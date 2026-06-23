"""
Phase 1 — Ingestion Verification

Goal: confirm the raw shot CSVs loaded from GCS are structurally sound
before any processing begins.

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

spark = (
    SparkSession.builder
    .appName("court-vision: shot data verification")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


RAW_PATH = "gs://pstat135-adam/raw/shots/NBA_*_Shots.csv"


shots = spark.read.csv(RAW_PATH, header=True, inferSchema=True)
shots.printSchema()


total_rows = shots.count()
print(f"\nTOTAL ROWS: {total_rows:,}")


print("\nSAMPLE ROWS (first 5):")
shots.show(5, truncate=False)


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


#Shot zone sanity check
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


#Three-point vs two-point split
if "SHOT_TYPE" in shots.columns:
    print("\nSHOT TYPE BREAKDOWN:")
    shots.groupBy("SHOT_TYPE").count().orderBy(F.desc("count")).show()


#Season coverage check
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


#Summary
print(f"\n{'='*60}")
print("  VERIFICATION COMPLETE")
print(f"  Total rows : {total_rows:,}")
print(f"  Columns    : {len(shots.columns)}")
print(f"  Source     : {RAW_PATH}")
print(f"{'='*60}\n")

spark.stop()