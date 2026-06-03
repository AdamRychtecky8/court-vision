"""
player_features.py — Phase 5 / Act 3, Part 1 of 2: PLAYER-SEASON FEATURE BUILD
==============================================================================
Aggregates ~4.2M shot rows up to ONE row per (player, season), described by WHERE that
player shot from: their share of attempts in each of the 6 court zones. That shot-location
fingerprint is what Part 2 (archetypes.py) clusters into player archetypes.

THE BIG IDEA
------------
Two players can take the same number of shots but be completely different players: one
lives at the rim, another lives behind the arc. Raw shot counts won't separate them, but
the DISTRIBUTION of their shots across zones will. So for each player-season we compute six
proportions (rim, paint, mid-range, left corner 3, right corner 3, above-the-break 3) that
sum to 1 — a compositional "shot profile."

HOW TO USE IT
-------------
- Run standalone (the __main__ block) to inspect the player-season table and verify the
  shares behave (they should sum to 1, the row count should look like real NBA rosters).
- Import from archetypes.py: call build_player_season() to get the table + feature vector.

HOW TO SUBMIT (standalone verification run; full data, so give it a minute):
   gcloud dataproc clusters start mycluster
   gcloud dataproc jobs submit pyspark 05_clustering/player_features.py --cluster=mycluster
   gcloud dataproc clusters stop mycluster
"""

from functools import reduce
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler

# ============================================================================
# CONFIG
# ============================================================================
INPUT_PATH = "gs://pstat135-adam/processed/shots/"

# Archetypes depend on per-player VOLUME, so we use the FULL dataset, not a sample.
# (Sampling shots would still estimate each player's shares, but it would shrink every
#  player's attempt count ~20x and make the MIN_ATTEMPTS filter meaningless.)
SAMPLE_FRACTION = None
SEED = 42

# Drop player-seasons below this many attempts: too few shots and the zone shares are
# noise. 200 keeps real rotation players and trims cup-of-coffee call-ups. ADJUSTABLE —
# watch the surviving row count in the diagnostics and lower it if too many players vanish
# (note: lockout 2012 and COVID 2020 had fewer games, so attempts run lower those years).
MIN_ATTEMPTS = 200

# Seasons >= this are the "post" (three-point) era; earlier seasons are "pre".
ERA_SPLIT = 2015

# The 6 zones (backcourt already removed in Phase 2) mapped to safe short names, in a
# FIXED order. The order here IS the order of the feature vector, which Part 2 relies on
# to interpret PCA loadings — do not reorder casually.
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
FEATURE_COLS = [b + "_share" for b in BASES]   # rim_share, paint_share, ... (vector order)


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

    # PIVOT: count attempts per (player, season) broken out into one column per zone.
    # Listing ZONES explicitly makes the pivot deterministic and skips an extra scan that
    # Spark would otherwise do just to discover the category values. Missing zone -> 0.
    pivoted = (shots.groupBy("PLAYER_ID", "PLAYER_NAME", "SEASON_1")
               .pivot("BASIC_ZONE", ZONES)
               .count()
               .fillna(0))

    # The pivot names columns after the raw zone strings (spaces/parens). Rename to the
    # safe short forms so later column references don't need backticks.
    for zone, base in ZONE_TO_BASE.items():
        pivoted = pivoted.withColumnRenamed(zone, base + "_cnt")

    # total_attempts = sum of the 6 zone counts. reduce(...) chains the six columns into
    # one addition expression (cleaner than a Python loop of withColumn calls).
    cnt_cols = [b + "_cnt" for b in BASES]
    total = reduce(lambda a, b: a + b, [F.col(c) for c in cnt_cols])
    players = pivoted.withColumn("total_attempts", total)

    # Noise filter: keep only player-seasons with enough shots for a stable profile.
    players = players.filter(F.col("total_attempts") >= MIN_ATTEMPTS)

    # Convert counts to shares (each zone's fraction of the player's total attempts).
    for base in BASES:
        players = players.withColumn(base + "_share", F.col(base + "_cnt") / F.col("total_attempts"))

    # Era label for the pre/post comparison in Part 2.
    players = players.withColumn(
        "ERA", F.when(F.col("SEASON_1") >= ERA_SPLIT, "post").otherwise("pre"))

    # Assemble the 6 shares into one vector named "features".
    players = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features").transform(players)

    return players


def main():
    spark = SparkSession.builder.appName("player_features").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    players = build_player_season(spark).cache()

    # --- How many player-seasons survived, and how they split by era ----------
    n = players.count()
    print(f"\n[build] player-seasons after MIN_ATTEMPTS>={MIN_ATTEMPTS}: {n:,}")
    print("[build] split by era (pre <2015 / post >=2015):")
    players.groupBy("ERA").count().orderBy("ERA").show()

    # --- Verify shares sum to 1 (a correctness check on the aggregation) -------
    # Floating-point means it won't be EXACTLY 1.0, but min and max should both be ~1.0.
    check = players.withColumn(
        "share_sum", reduce(lambda a, b: a + b, [F.col(c) for c in FEATURE_COLS]))
    print("[verify] share_sum should be ~1.0 for every row:")
    check.select(F.min("share_sum").alias("min"), F.max("share_sum").alias("max")).show()

    # --- Attempt-count distribution (did the filter behave sensibly?) ----------
    print("[verify] total_attempts distribution:")
    players.select("total_attempts").summary("min", "25%", "50%", "75%", "max").show()

    # --- League-average shot profile (a sanity check on the shares themselves) -
    # Across all player-seasons, the mean shares should look like real NBA shot selection:
    # the rim and above-the-break 3 are the big buckets; the corners are small.
    print("[verify] mean zone shares across all player-seasons:")
    players.select(*[F.mean(c).alias(c) for c in FEATURE_COLS]).show()

    # --- A few recognizable examples ------------------------------------------
    # High-volume recent seasons: eyeball whether the profile matches the player you know.
    print("[verify] sample of high-volume player-seasons:")
    (players.filter(F.col("SEASON_1") >= 2022)
            .orderBy(F.desc("total_attempts"))
            .select("PLAYER_NAME", "SEASON_1", "total_attempts", *FEATURE_COLS)
            .show(10, truncate=False))

    spark.stop()


if __name__ == "__main__":
    main()