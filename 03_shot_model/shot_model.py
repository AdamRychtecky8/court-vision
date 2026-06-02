"""
shot_model.py
=============
Shot-make model — Part 2 of 2: MODELING + EVALUATION.
Pairs with feature_engineering.py (Part 1), which it imports.

WHAT THIS FILE DOES
-------------------
1. Loads the cleaned shot Parquet, builds ACTION_GROUP, and assembles the 26-length
   feature vector using the SAME stages from Part 1 (no duplication).
2. Splits into train/test with a fixed seed.
3. LASSO REGULARIZATION PATH: fits an L1 logistic regression at several penalty
   strengths to SELECT and RANK features by the order they survive — which shot
   characteristics actually predict a make (Act 1's evidence).
4. Two full classifiers — LogisticRegression and LinearSVC — trained and compared on
   ROC-AUC, F1, accuracy, plus a confusion matrix.
5. Saves the LASSO path (Table 3a), the headline coefficients (Table 3), and the
   model-comparison table (Table 4) to GCS for the report.

HOW TO SUBMIT  (--py-files ships Part 1 to the cluster as a dependency):
   gcloud dataproc clusters start mycluster
   gcloud dataproc jobs submit pyspark 03_shot_model/shot_model.py `
       --cluster=mycluster `
       --py-files=03_shot_model/feature_engineering.py
   gcloud dataproc clusters stop mycluster

WHY A PATH INSTEAD OF ONE PENALTY VALUE
---------------------------------------
A single regParam is fragile: too strong and LASSO collapses many collinear location
features onto one proxy (hiding the zone-by-zone story); too weak and it selects
nothing out. Sweeping a range shows the ORDER features enter the model as the penalty
relaxes. That ordering is a stable importance ranking even when features are
correlated, and the fact that the CONTEXT features (clutch, quarter, clock) stay at
zero across the whole path is itself the "location dominates context" finding.
"""

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, LinearSVC
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)

from feature_engineering import (
    add_action_group,
    build_feature_stages,
    INPUT_PATH,
    LABEL_COL,
    SAMPLE_FRACTION,
    SEED,
)

# LASSO penalty strengths to sweep, strong -> weak. Add/remove values freely.
REG_PATH = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
# The penalty whose coefficients become the headline Table 3. Pick one from the path
# output that keeps the zone structure without selecting everything back in.
HEADLINE_REG = 0.001

LASSO_PATH_OUT = "gs://pstat135-adam/processed/tables/lasso_path"
LASSO_OUT = "gs://pstat135-adam/processed/tables/lasso_coefficients"
COMPARISON_OUT = "gs://pstat135-adam/processed/tables/model_comparison"


def extract_feature_names(df, features_col="features"):
    """
    Map each slot of the 26-length feature vector back to a readable name, by reading
    the ML metadata Spark attaches to the assembled column. Without this, coefficients
    are just an anonymous list of 26 numbers. Numeric slots keep names like LOC_X;
    one-hot slots are named after the category they flag.
    """
    attrs = df.schema[features_col].metadata["ml_attr"]["attrs"]
    indexed = []
    for group in attrs.values():       # groups like "numeric" / "binary"
        for a in group:
            indexed.append((a["idx"], a["name"]))
    indexed.sort()                     # by slot index -> lines up with coefficient array
    return [name for _, name in indexed]


def print_confusion_matrix(predictions, label_col=LABEL_COL):
    """
    2x2 confusion matrix + precision/recall. Rows = actual, cols = predicted:
        TN called miss/was miss   FP called make/was miss
        FN called miss/was make   TP called make/was make
    Precision and recall expose class-specific weakness that overall accuracy hides.
    """
    rows = (predictions.groupBy(label_col, "prediction").count()
            .orderBy(label_col, "prediction").collect())
    c = {(int(r[label_col]), int(r["prediction"])): r["count"] for r in rows}
    tn, fp, fn, tp = c.get((0, 0), 0), c.get((0, 1), 0), c.get((1, 0), 0), c.get((1, 1), 0)
    print("                  predicted miss    predicted make")
    print(f"   actual miss   {tn:>14,}   {fp:>15,}")
    print(f"   actual make   {fn:>14,}   {tp:>15,}")
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    print(f"   precision (of predicted makes, share truly made): {precision:.3f}")
    print(f"   recall    (of actual makes, share we caught):     {recall:.3f}")


def main():
    spark = SparkSession.builder.appName("shot_model").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    # --- Load + ACTION_GROUP, as in Part 1 ------------------------------------
    shots = spark.read.parquet(INPUT_PATH)
    if SAMPLE_FRACTION is not None:
        shots = shots.sample(fraction=SAMPLE_FRACTION, seed=SEED)
    shots = add_action_group(shots)

    # --- Train/test split BEFORE fitting transformers (leakage discipline) ----
    train, test = shots.randomSplit([0.8, 0.2], seed=SEED)
    train.cache(); test.cache()
    print(f"\n[split] train rows: {train.count():,} | test rows: {test.count():,}")

    # --- Fit feature pipeline on TRAIN, apply to both splits ------------------
    feature_model = Pipeline(stages=build_feature_stages()).fit(train)
    train_feat = feature_model.transform(train).select("features", LABEL_COL).cache()
    test_feat = feature_model.transform(test).select("features", LABEL_COL).cache()
    print(f"[features] vector length: {train_feat.first()['features'].size}")

    feature_names = extract_feature_names(train_feat)

    # ========================================================================
    # STEP 1 — LASSO regularization path (the Act 1 evidence)
    # ========================================================================
    # elasticNetParam=1.0 -> pure L1 (LASSO). We fit it at each penalty in REG_PATH.
    # standardization=True (default) rescales features before penalizing so units
    # don't bias selection; coefficients are reported back on the original scale.
    print("\n=== STEP 1: LASSO regularization path ===")
    print("  How many features survive at each penalty (strong -> weak):")
    path_rows = []           # long-format (regParam, feature, coefficient) for the figure
    headline_coefs = None
    for rp in REG_PATH:
        m = LogisticRegression(
            featuresCol="features", labelCol=LABEL_COL,
            elasticNetParam=1.0, regParam=rp, maxIter=50,
        ).fit(train_feat)
        arr = m.coefficients.toArray()
        kept = int((arr != 0).sum())
        print(f"    regParam={rp:<8}  kept {kept:>2}/{len(arr)} features")
        for name, coef in zip(feature_names, arr):
            path_rows.append((float(rp), name, float(coef)))
        if rp == HEADLINE_REG:
            headline_coefs = list(zip(feature_names, arr))

    # Headline coefficients (at HEADLINE_REG), ranked by magnitude -> report Table 3.
    headline_sorted = sorted(headline_coefs, key=lambda x: abs(x[1]), reverse=True)
    print(f"\n  Coefficients at regParam={HEADLINE_REG} (ranked by magnitude):")
    print("  feature                                            coefficient")
    for name, coef in headline_sorted:
        flag = "   <- dropped" if coef == 0.0 else ""
        print(f"  {name:<48} {coef:>10.4f}{flag}")
    n_zero = sum(1 for _, coef in headline_coefs if coef == 0.0)
    print(f"  At regParam={HEADLINE_REG}: kept {len(headline_coefs) - n_zero} "
          f"of {len(headline_coefs)} features.")

    # ========================================================================
    # STEP 2 — Full classifiers + evaluation
    # ========================================================================
    # These use ALL features (the selector above is for interpretation, not for
    # pruning the inputs). Expected AUC is low-to-mid 0.6s: location + action carry
    # real but limited signal, and with no defender data that's the honest ceiling.
    print("\n=== STEP 2: train + evaluate classifiers ===")
    logreg = LogisticRegression(
        featuresCol="features", labelCol=LABEL_COL,
        regParam=0.01, elasticNetParam=0.0, maxIter=50,   # light L2 for stability
    )
    svc = LinearSVC(
        featuresCol="features", labelCol=LABEL_COL,
        regParam=0.01, maxIter=50,                         # max-margin linear classifier
    )

    auc_eval = BinaryClassificationEvaluator(
        labelCol=LABEL_COL, rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    f1_eval = MulticlassClassificationEvaluator(
        labelCol=LABEL_COL, predictionCol="prediction", metricName="f1")
    acc_eval = MulticlassClassificationEvaluator(
        labelCol=LABEL_COL, predictionCol="prediction", metricName="accuracy")

    results = []
    for name, model in [("LogisticRegression", logreg), ("LinearSVC", svc)]:
        fitted = model.fit(train_feat)
        preds = fitted.transform(test_feat)
        auc, f1, acc = auc_eval.evaluate(preds), f1_eval.evaluate(preds), acc_eval.evaluate(preds)
        results.append((name, auc, f1, acc))
        print(f"\n  --- {name} ---")
        print(f"  ROC-AUC : {auc:.4f}   (0.5 = coin flip; ~0.60-0.70 is the location-only ceiling)")
        print(f"  F1      : {f1:.4f}")
        print(f"  Accuracy: {acc:.4f}   (always-guess-miss floor is ~0.54)")
        print_confusion_matrix(preds)

    # ========================================================================
    # STEP 3 — Save report tables to GCS
    # ========================================================================
    print("\n=== STEP 3: save tables to GCS ===")
    # Table 3a — full path (regParam, feature, coefficient) — data for the path figure.
    spark.createDataFrame(path_rows, ["regParam", "feature", "coefficient"]) \
        .coalesce(1).write.mode("overwrite").option("header", True).csv(LASSO_PATH_OUT)
    print(f"  wrote {LASSO_PATH_OUT}")

    # Table 3 — headline coefficients ranked by magnitude.
    spark.createDataFrame(
        [(n, float(c)) for n, c in headline_sorted], ["feature", "coefficient"]) \
        .coalesce(1).write.mode("overwrite").option("header", True).csv(LASSO_OUT)
    print(f"  wrote {LASSO_OUT}")

    # Table 4 — model comparison.
    spark.createDataFrame(
        [(n, float(a), float(f), float(ac)) for n, a, f, ac in results],
        ["model", "roc_auc", "f1", "accuracy"]) \
        .coalesce(1).write.mode("overwrite").option("header", True).csv(COMPARISON_OUT)
    print(f"  wrote {COMPARISON_OUT}")

    spark.stop()


if __name__ == "__main__":
    main()