"""
Phase 2 — Processing & Feature Engineering

Goal: take the raw CSVs from GCS, clean them, add engineered features,
and write the result as Parquet partitioned by season.

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

spark = (
    SparkSession.builder
    .appName("court-vision: shot data processing")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

RAW_PATH       = "gs://pstat135-adam/raw/shots/NBA_*_Shots.csv"
PROCESSED_PATH = "gs://pstat135-adam/processed/shots/"
TABLES_PATH    = "gs://pstat135-adam/processed/tables/"

shots_raw = spark.read.csv(RAW_PATH, header=True, inferSchema=True)


shots_clean = (
    shots_raw

    # SHOT_MADE boolean to integer as SHOT_MADE_INT new column
    #GAME_DATE string to date type as GAME_DATE_PARSED new column
    .withColumn("SHOT_MADE_INT", F.col("SHOT_MADE").cast("integer")) 
    .withColumn("GAME_DATE_PARSED", F.to_date(F.col("GAME_DATE"), "MM-dd-yyyy")) 
)


#Feature engineering
shots_featured = (
    shots_clean

    # SECS_REMAINING: total seconds left in the current quarter.
    # Merges MINS_LEFT and SECS_LEFT into one clean continuous number.
    .withColumn(
        "SECS_REMAINING",
        F.col("MINS_LEFT") * 60 + F.col("SECS_LEFT")
    )

    # IS_3PT: 1 if this is a 3-point attempt, 0 if 2-point.
    .withColumn(
        "IS_3PT",
        F.when(F.col("SHOT_TYPE") == "3PT Field Goal", 1).otherwise(0)
    )

    # SHOT_VALUE: how many points this attempt is worth if made (2 or 3).
    .withColumn(
        "SHOT_VALUE",
        F.when(F.col("SHOT_TYPE") == "3PT Field Goal", 3).otherwise(2)
    )

    # IS_CLUTCH: 1 if this shot was taken in the final 2 minutes of
    # the 4th quarter or any overtime period, 0 otherwise.
    .withColumn(
        "IS_CLUTCH",
        F.when(
            (F.col("QUARTER") >= 4) & (F.col("SECS_REMAINING") <= 120), 1
        ).otherwise(0)
    )

    # EXPECTED_PTS: actual points scored on this specific attempt.
    .withColumn(
        "EXPECTED_PTS",
        F.col("SHOT_VALUE") * F.col("SHOT_MADE_INT")
    )
)


#Filter out backcourt shots outliers for the purpose of this analysis

backcourt_count = shots_featured.filter(F.col("BASIC_ZONE") == "Backcourt").count()
shots_processed = shots_featured.filter(F.col("BASIC_ZONE") != "Backcourt")


#FG% and efficiency by zone — Table 1 for the report

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

fg_by_zone.show(truncate=False)

(
    fg_by_zone
    .coalesce(1)
    .write
    .mode("overwrite")
    .option("header", "true")
    .csv(TABLES_PATH + "fg_pct_by_zone/")
)


#3PT rate by season — era analysis preview

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

three_pt_by_season.show(25, truncate=False)

(
    three_pt_by_season
    .coalesce(1)
    .write
    .mode("overwrite")
    .option("header", "true")
    .csv(TABLES_PATH + "three_pt_rate_by_season/")
)
print(f"  Saved to: {TABLES_PATH}three_pt_rate_by_season/\n")


(
    shots_processed
    .write
    .mode("overwrite")
    .partitionBy("SEASON_1")
    .parquet(PROCESSED_PATH)
)

print("Parquet write complete.\n")


#Verify the Parquet output

shots_verify = spark.read.parquet(PROCESSED_PATH)

verified_rows = shots_verify.count()
print(f"\nPARQUET VERIFICATION:")
print(f"  Rows written     : {verified_rows:,}")
print(f"  Columns          : {len(shots_verify.columns)}")
print(f"  Location         : {PROCESSED_PATH}")

shots_verify.printSchema()

shots_verify.groupBy("SEASON_1").count().orderBy("SEASON_1").show(25)


#Summary
print("  PHASE 2 COMPLETE")
print(f"  Rows in Parquet  : {verified_rows:,}")
print(f"  Features added   : SHOT_MADE_INT, GAME_DATE_PARSED,")
print(f"                     SECS_REMAINING, IS_3PT, SHOT_VALUE,")
print(f"                     IS_CLUTCH, EXPECTED_PTS")
print(f"  Tables saved     : fg_pct_by_zone, three_pt_rate_by_season")
print(f"  Parquet path     : {PROCESSED_PATH}")
print(f"")

spark.stop()
