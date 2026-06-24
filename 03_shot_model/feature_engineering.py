from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler

INPUT_PATH = "gs://pstat135-adam/processed/shots/"
SAMPLE_FRACTION = None #0.05 in development; set to None for full dataset
SEED = 42
LABEL_COL = "SHOT_MADE_INT"

NUMERIC_COLS = [
    "LOC_X",           # horizontal court coordinate
    "LOC_Y",           # vertical court coordinate
    "SHOT_DISTANCE",   # feet from the hoop
    "SECS_REMAINING",  # seconds left in the quarter (engineered in Phase 2)
    "IS_3PT",          # 1 if a three, 0 if a two
    "SHOT_VALUE",      # 2 or 3 — point value of the shot
    "IS_CLUTCH",       # 1 if Q4/OT with <=120s left
    "QUARTER",         # 1-4 (5+ = OT)
]

CATEGORICAL_COLS = [
    "BASIC_ZONE",    # 6 court zones (Restricted Area, Mid-Range, corners, etc.)
    "SHOT_TYPE",     # "2PT Field Goal" / "3PT Field Goal"
    "ZONE_RANGE",    # distance band
    "ACTION_GROUP",  # bucketed ACTION_TYPE (built in add_action_group)
]


def add_action_group(df):
    """
    Collapse the many raw ACTION_TYPE values ("Driving Layup Shot",
    "Step Back Jump shot", "Turnaround Hook Shot") into 5 broad buckets.
    """
    lc = F.lower(F.col("ACTION_TYPE"))
    action_group = (
        F.when(lc.contains("dunk"), "Dunk")     # at the rim, above it: Driving/Alley Oop/Cutting Dunk
         .when(lc.contains("layup"), "Layup")   # at the rim, finesse: Driving/Reverse/Finger Roll Layup
         .when(lc.contains("hook"), "Hook")     # post move: Hook Shot, Turnaround Hook Shot
         .when(
             lc.contains("jump")        # "Jump Shot", "Jumper", "Running Jump Shot"
             | lc.contains("fade")      # "Fadeaway", "Turnaround Fadeaway"
             | lc.contains("pullup")    # "Pullup Jump shot"
             | lc.contains("pull-up")
             | lc.contains("step back") # "Step Back Jump shot"
             | lc.contains("stepback")
             | lc.contains("turnaround")
             | lc.contains("floating"), # "Floating Jump shot"
             "Jump Shot",
         )
         .otherwise("Other")            # Tip Shot, Putback, Bank Shot, anything unmatched
    )
    return df.withColumn("ACTION_GROUP", action_group)


def build_feature_stages():
    """
    Build the ordered list of transformer STAGES that turn raw columns into one
    "features" vector. Returns the list (not a fitted model) so shot_model.py can
    append a classifier and let a single Pipeline fit everything on the TRAINING
    split only, which is what prevents test-set information from leaking in.
    """
    
    # 1) one indexer per categorical column:  COL -> COL_idx
    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        for c in CATEGORICAL_COLS
    ]

    # 2) a single encoder handling all the index columns at once:  COL_idx -> COL_ohe
    encoder = OneHotEncoder(
        inputCols=[f"{c}_idx" for c in CATEGORICAL_COLS],
        outputCols=[f"{c}_ohe" for c in CATEGORICAL_COLS],
    )

    # 3) assemble numeric + one-hot columns into the final "features" vector
    assembler = VectorAssembler(
        inputCols=NUMERIC_COLS + [f"{c}_ohe" for c in CATEGORICAL_COLS],
        outputCol="features",
        handleInvalid="skip",
    )
    
    return indexers + [encoder, assembler]


def main():
    
    spark = SparkSession.builder.appName("shot_feature_engineering").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    shots = spark.read.parquet(INPUT_PATH)
    
    if SAMPLE_FRACTION is not None:
        shots = shots.sample(fraction=SAMPLE_FRACTION, seed=SEED)

    shots = add_action_group(shots)
    shots.cache()

    print(f"\n[load] rows after sampling: {shots.count():,}")

    shots.groupBy("ACTION_TYPE").count().orderBy(F.desc("count")).show(30, truncate=False)
    
    print("[diagnostic] ACTION_GROUP bucket sizes:")
    shots.groupBy("ACTION_GROUP").count().orderBy(F.desc("count")).show(truncate=False)

    print("[diagnostic] label balance (SHOT_MADE_INT):")
    shots.groupBy(LABEL_COL).count().orderBy(LABEL_COL).show()

    stages = build_feature_stages()
    feature_pipeline = Pipeline(stages=stages).fit(shots)
    transformed = feature_pipeline.transform(shots)

    first_row = transformed.select("features").first()
    vector_size = first_row["features"].size
    print(f"\n[verify] assembled feature vector length = {vector_size}")
    print(f"[verify] numeric features contributed: {len(NUMERIC_COLS)}")
    print("[verify] distinct categories per categorical column:")
    for c in CATEGORICAL_COLS:
        n = transformed.select(c).distinct().count()
        print(f"         {c}: {n}")

    print("\n[verify] sample rows (categoricals + label + assembled features):")
    transformed.select(*CATEGORICAL_COLS, LABEL_COL, "features").show(5, truncate=60)
    
    print(f"[done] rows available for modeling: {transformed.count():,}")

    spark.stop()


if __name__ == "__main__":
    main()