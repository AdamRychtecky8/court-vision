"""
Phase 5 / Act 3, Part 2 of 2: PCA + K-MEANS PLAYER ARCHETYPES

1. Build the player-season shot fingerprints (6 zone shares) from Part 1.
2. Scale every feature before PCA/clustering.
3. PCA — find the axes of variation in shot selection, and read the loadings to interpret them.
4. Silhouette sweep and k value selection
5. K-Means at the chosen k, then name the clusters by their average shot profiles (the "archetypes").
6. Era comparison: hold the archetype definitions fixed and compare how the
   POPULATION of each archetype shifted from the pre-2015 era to the post-2015 era.

Bash Commands:
   gcloud dataproc clusters start mycluster
   gcloud dataproc jobs submit pyspark 05_clustering/archetypes.py `
       --cluster=mycluster `
       --py-files=05_clustering/player_features.py
   gcloud dataproc clusters stop mycluster
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import StandardScaler, PCA
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.functions import vector_to_array
from player_features import build_player_season, FEATURE_COLS, SEED

N_PCA = 5                       # keep 5 of 6 components
K_RANGE = [3, 4, 5, 6, 7, 8]    # various k values
CHOSEN_K = 5                    # final k revisited in post

PROFILE_OUT = "gs://pstat135-adam/processed/tables/cluster_profiles"
ERA_OUT = "gs://pstat135-adam/processed/tables/cluster_by_era"
LOADINGS_OUT = "gs://pstat135-adam/processed/tables/pca_loadings"
ASSIGN_OUT = "gs://pstat135-adam/processed/tables/player_clusters"  


def main():
    spark = SparkSession.builder.appName("archetypes").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    players = build_player_season(spark)

    scaler = StandardScaler(inputCol="features", outputCol="scaled",
                            withMean=True, withStd=True)
    players = scaler.fit(players).transform(players)

    pca_model = PCA(k=N_PCA, inputCol="scaled", outputCol="pca_features").fit(players)
    players = pca_model.transform(players).cache()

    ev = pca_model.explainedVariance.toArray()
    for j, frac in enumerate(ev):
        print(f"    PC{j+1}: {frac:6.1%}   (cumulative {ev[:j+1].sum():6.1%})")

    pc = pca_model.pc.toArray()
    header = "    " + "feature".ljust(12) + "".join(f"PC{j+1:>8}" for j in range(N_PCA))
    print(header)
    for i, fname in enumerate(FEATURE_COLS):
        print("    " + fname.ljust(12) + "".join(f"{pc[i][j]:>9.3f}" for j in range(N_PCA)))

    evaluator = ClusteringEvaluator(featuresCol="pca_features",
                                    predictionCol="prediction", metricName="silhouette")
    for k in K_RANGE:
        km = KMeans(featuresCol="pca_features", predictionCol="prediction", k=k, seed=SEED)
        pred = km.fit(players).transform(players)
        print(f"    k={k}: silhouette = {evaluator.evaluate(pred):.4f}")


    km_model = KMeans(featuresCol="pca_features", predictionCol="cluster",
                      k=CHOSEN_K, seed=SEED).fit(players)
    assigned = km_model.transform(players).cache()

    profile = (assigned.groupBy("cluster")
               .agg(F.count("*").alias("n"),
                    *[F.round(F.mean(c), 3).alias(c) for c in FEATURE_COLS])
               .orderBy("cluster"))
    profile.show(truncate=False)

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

    profile.coalesce(1).write.mode("overwrite").option("header", True).csv(PROFILE_OUT)
    era_tbl.coalesce(1).write.mode("overwrite").option("header", True).csv(ERA_OUT)

    # PCA loadings as a tidy table.
    loadings_rows = [tuple([FEATURE_COLS[i]] + [float(pc[i][j]) for j in range(N_PCA)])
                     for i in range(len(FEATURE_COLS))]
    spark.createDataFrame(loadings_rows, ["feature"] + [f"PC{j+1}" for j in range(N_PCA)]) \
        .coalesce(1).write.mode("overwrite").option("header", True).csv(LOADINGS_OUT)

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