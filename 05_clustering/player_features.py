"""
Phase 5 / Act 3, Part 1 of 2: PLAYER-SEASON FEATURE BUILD
==============================================================================
Aggregates ~4.2M shot rows up to ONE row per (player, season), described by WHERE that
player shot from: their share of attempts in each of the 6 court zones. That shot-location
fingerprint is what Part 2 (archetypes.py) clusters into player archetypes.

Bash Commands:
   gcloud dataproc clusters start mycluster
   gcloud dataproc jobs submit pyspark 05_clustering/player_features.py --cluster=mycluster
   gcloud dataproc clusters stop mycluster
"""

from functools import reduce
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler

INPUT_PATH = "gs://pstat135-adam/processed/shots/"

SAMPLE_FRACTION = None
SEED = 42

# Drop player-seasons below this many attempts
MIN_ATTEMPTS = 200

# Seasons >= this are the "post" (three-point) era; earlier seasons are "pre".
ERA_SPLIT = 2015

# The 6 zones (backcourt already removed in Phase 2) mapped to safe short names, in a FIXED order. 
ZONE_TO_BASE = {
    "Restricted Area": "rim",
    "In The Paint (Non-RA)": "paint",
    "Mid-Range": "mid",
    "Left Corner 3": "lc3",
    "Right Corner 3": "rc3",
    "Above the Break 3": "atb3",
}
ZONES = list(ZONE_TO_BASE.keys())
BASES = list(ZONE_TO_BASE.values())
FEATURE_COLS = [b + "_share" for b in BASES]


def build_player_season(spark):
    """
    Returns one row per (PLAYER_ID, PLAYER_NAME, SEASON_1) with:
      - total_attempts
      - the 6 zone-share columns (FEATURE_COLS), which sum to 1
      - ERA ("pre"/"post")
      - an assembled "features" vector of the 6 shares (raw — Part 2 standardizes it)
    """
    shots = spark.read.parquet(INPUT_PATH)
    if SAMPLE_FRACTION is not None:
        shots = shots.sample(fraction=SAMPLE_FRACTION, seed=SEED)

    pivoted = (shots.groupBy("PLAYER_ID", "PLAYER_NAME", "SEASON_1")
               .pivot("BASIC_ZONE", ZONES)
               .count()
               .fillna(0))

    for zone, base in ZONE_TO_BASE.items():
        pivoted = pivoted.withColumnRenamed(zone, base + "_cnt")

    cnt_cols = [b + "_cnt" for b in BASES]
    total = reduce(lambda a, b: a + b, [F.col(c) for c in cnt_cols])
    players = pivoted.withColumn("total_attempts", total)

    # Noise filter: keep only player-seasons with enough shots for a stable profile.
    players = players.filter(F.col("total_attempts") >= MIN_ATTEMPTS)

    for base in BASES:
        players = players.withColumn(base + "_share", F.col(base + "_cnt") / F.col("total_attempts"))

    # Era label
    players = players.withColumn(
        "ERA", F.when(F.col("SEASON_1") >= ERA_SPLIT, "post").otherwise("pre"))

    players = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features").transform(players)

    return players


def main():
    spark = SparkSession.builder.appName("player_features").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    players = build_player_season(spark).cache()

    n = players.count()
    players.groupBy("ERA").count().orderBy("ERA").show()

    # Verify shares sum to 1
    check = players.withColumn(
        "share_sum", reduce(lambda a, b: a + b, [F.col(c) for c in FEATURE_COLS]))
    check.select(F.min("share_sum").alias("min"), F.max("share_sum").alias("max")).show()

    players.select("total_attempts").summary("min", "25%", "50%", "75%", "max").show()

    # Sanity check: the mean of each zone-share column as expected
    players.select(*[F.mean(c).alias(c) for c in FEATURE_COLS]).show()

    #A few recognizable examples by eye
    (players.filter(F.col("SEASON_1") >= 2022)
            .orderBy(F.desc("total_attempts"))
            .select("PLAYER_NAME", "SEASON_1", "total_attempts", *FEATURE_COLS)
            .show(10, truncate=False))

    spark.stop()


if __name__ == "__main__":
    main()