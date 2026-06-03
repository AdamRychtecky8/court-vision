"""
archetypes.py — Phase 5 / Act 3, Part 2 of 2: PCA + K-MEANS PLAYER ARCHETYPES
=============================================================================
Pairs with player_features.py (Part 1), which it imports.

THE PLAN
--------
1. Build the player-season shot fingerprints (6 zone shares) from Part 1.
2. StandardScaler — put every zone share on the same footing before PCA/clustering.
3. PCA — find the dominant AXES of variation in shot selection. We expect the first axis
   to be a "rim vs. perimeter" contrast; reading the loadings tells us what each axis means.
4. Silhouette sweep — try several values of k (number of archetypes) and let the data hint
   at how many natural groups there are.
5. K-Means at the chosen k — assign every player-season to an archetype, then read each
   cluster's average shot profile to NAME the archetypes.
6. Era comparison — the payoff: hold the archetype definitions fixed and compare how the
   POPULATION of each archetype shifted from the pre-2015 era to the post-2015 era. If the
   mid-range archetype shrank while a perimeter archetype grew, that is the role
   reorganization, shown with numbers.

HOW TO SUBMIT (--py-files ships Part 1 to the cluster):
   gcloud dataproc clusters start mycluster
   gcloud dataproc jobs submit pyspark 05_clustering/archetypes.py `
       --cluster=mycluster `
       --py-files=05_clustering/player_features.py
   gcloud dataproc clusters stop mycluster

WHY PCA BEFORE K-MEANS HERE
---------------------------
Two reasons. (1) The six shares are CORRELATED (a player who shoots more threes
mechanically shoots fewer twos), and PCA turns them into uncorrelated axes that K-Means
handles more cleanly. (2) Because the six shares SUM TO 1, one direction in the data
carries essentially no real variation — its PCA explained-variance comes out near zero.
Clustering on the standardized shares directly would rescale that near-empty direction up
to unit variance and amplify rounding noise; dropping it via PCA avoids that. We keep the
top 5 components (the meaningful ones).
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import StandardScaler, PCA
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.functions import vector_to_array

# Borrow the player-season builder + the feature list/constants from Part 1.
from player_features import build_player_season, FEATURE_COLS, SEED

N_PCA = 5                       # keep 5 of 6 components (the 6th is the ~empty compositional direction)
K_RANGE = [3, 4, 5, 6, 7, 8]    # candidate archetype counts to score
CHOSEN_K = 5                    # final k. Revisit after reading the silhouette sweep + profiles.

PROFILE_OUT = "gs://pstat135-adam/processed/tables/cluster_profiles"
ERA_OUT = "gs://pstat135-adam/processed/tables/cluster_by_era"
LOADINGS_OUT = "gs://pstat135-adam/processed/tables/pca_loadings"
ASSIGN_OUT = "gs://pstat135-adam/processed/tables/player_clusters"   # for the Figure 5 scatter


def main():
    spark = SparkSession.builder.appName("archetypes").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    players = build_player_season(spark)

    # ========================================================================
    # STEP 1 — Standardize the 6 zone shares
    # ========================================================================
    # withMean+withStd: each share becomes mean 0, standard deviation 1. PCA is
    # scale-sensitive, and without this the higher-variance zones (rim, mid) would
    # dominate the lower-variance corner zones purely by spread.
    scaler = StandardScaler(inputCol="features", outputCol="scaled",
                            withMean=True, withStd=True)
    players = scaler.fit(players).transform(players)

    # ========================================================================
    # STEP 2 — PCA: find and interpret the axes of shot selection
    # ========================================================================
    pca_model = PCA(k=N_PCA, inputCol="scaled", outputCol="pca_features").fit(players)
    players = pca_model.transform(players).cache()

    ev = pca_model.explainedVariance.toArray()
    print("\n=== STEP 2: PCA ===")
    print("  explained variance per component (should sum to ~1.0; one direction is ~empty):")
    for j, frac in enumerate(ev):
        print(f"    PC{j+1}: {frac:6.1%}   (cumulative {ev[:j+1].sum():6.1%})")

    # Loadings: pca_model.pc is a (6 features x N_PCA) matrix. Each column is a PC; the
    # entries say how much each original zone share contributes to that axis. Reading PC1's
    # signs tells you what the main axis means (e.g. positive on 3s, negative on rim = a
    # perimeter-vs-rim axis).
    pc = pca_model.pc.toArray()
    print("\n  loadings (rows = zone share, cols = principal components):")
    header = "    " + "feature".ljust(12) + "".join(f"PC{j+1:>8}" for j in range(N_PCA))
    print(header)
    for i, fname in enumerate(FEATURE_COLS):
        print("    " + fname.ljust(12) + "".join(f"{pc[i][j]:>9.3f}" for j in range(N_PCA)))

    # ========================================================================
    # STEP 3 — Silhouette sweep: how many archetypes?
    # ========================================================================
    # Silhouette ranges -1..1; higher = tighter, better-separated clusters. There's rarely a
    # single "correct" k — use this together with whether the profiles in STEP 5 are
    # interpretable. Real basketball roles don't always give a sharp silhouette peak.
    print("\n=== STEP 3: silhouette by k (on PCA features) ===")
    evaluator = ClusteringEvaluator(featuresCol="pca_features",
                                    predictionCol="prediction", metricName="silhouette")
    for k in K_RANGE:
        km = KMeans(featuresCol="pca_features", predictionCol="prediction", k=k, seed=SEED)
        pred = km.fit(players).transform(players)
        print(f"    k={k}: silhouette = {evaluator.evaluate(pred):.4f}")

    # ========================================================================
    # STEP 4 — Final K-Means at CHOSEN_K
    # ========================================================================
    print(f"\n=== STEP 4: K-Means at k={CHOSEN_K} ===")
    km_model = KMeans(featuresCol="pca_features", predictionCol="cluster",
                      k=CHOSEN_K, seed=SEED).fit(players)
    assigned = km_model.transform(players).cache()

    # ========================================================================
    # STEP 5 — Name the archetypes from their average shot profiles
    # ========================================================================
    # Each cluster's mean zone shares describe its prototypical player. Reading these is how
    # you label clusters by hand: e.g. high rim + near-zero 3s = "rim-running big";
    # high atb3 = "perimeter shooter"; high mid = "mid-range scorer".
    print("\n=== STEP 5: cluster profiles (mean zone share per archetype) ===")
    profile = (assigned.groupBy("cluster")
               .agg(F.count("*").alias("n"),
                    *[F.round(F.mean(c), 3).alias(c) for c in FEATURE_COLS])
               .orderBy("cluster"))
    profile.show(truncate=False)

    # ========================================================================
    # STEP 6 — Era comparison (the Act 3 finding)
    # ========================================================================
    # Hold the archetype definitions fixed and ask: what share of PRE-2015 player-seasons
    # fell in each cluster, versus POST-2015? The deltas show roles reorganizing.
    print("\n=== STEP 6: archetype population by era ===")
    era_totals = {r["ERA"]: r["count"]
                  for r in assigned.groupBy("ERA").count().collect()}
    era_tbl = (assigned.groupBy("cluster").pivot("ERA", ["pre", "post"])
               .count().fillna(0)
               .withColumn("pre_pct", F.round(F.col("pre") / era_totals["pre"] * 100, 1))
               .withColumn("post_pct", F.round(F.col("post") / era_totals["post"] * 100, 1))
               .withColumn("delta_pct", F.round(F.col("post_pct") - F.col("pre_pct"), 1))
               .orderBy("cluster"))
    print("  (pre/post = raw counts; *_pct = % of that era's player-seasons; delta = post - pre)")
    era_tbl.show(truncate=False)

    # ========================================================================
    # STEP 7 — Save tables to GCS (report + Figure 5)
    # ========================================================================
    print("\n=== STEP 7: save tables to GCS ===")
    profile.coalesce(1).write.mode("overwrite").option("header", True).csv(PROFILE_OUT)
    era_tbl.coalesce(1).write.mode("overwrite").option("header", True).csv(ERA_OUT)

    # PCA loadings as a tidy table.
    loadings_rows = [tuple([FEATURE_COLS[i]] + [float(pc[i][j]) for j in range(N_PCA)])
                     for i in range(len(FEATURE_COLS))]
    spark.createDataFrame(loadings_rows, ["feature"] + [f"PC{j+1}" for j in range(N_PCA)]) \
        .coalesce(1).write.mode("overwrite").option("header", True).csv(LOADINGS_OUT)

    # Per-player-season assignments + first two PC scores, for the local 2D scatter (Fig 5).
    out = (assigned
           .withColumn("pca_arr", vector_to_array("pca_features"))
           .withColumn("PC1", F.col("pca_arr")[0])
           .withColumn("PC2", F.col("pca_arr")[1])
           .select("PLAYER_NAME", "SEASON_1", "ERA", "cluster", "PC1", "PC2", *FEATURE_COLS))
    out.coalesce(1).write.mode("overwrite").option("header", True).csv(ASSIGN_OUT)
    for path in (PROFILE_OUT, ERA_OUT, LOADINGS_OUT, ASSIGN_OUT):
        print(f"  wrote {path}")

    spark.stop()


if __name__ == "__main__":
    main()