"""
feature_engineering.py
=======================
Shot-make model — Part 1 of 2: FEATURE ENGINEERING.
(Phase 4 in master-guide order. Part 2 is shot_model.py, written next.)

WHAT THIS FILE DOES, IN ONE SENTENCE
------------------------------------
It turns the cleaned shot Parquet into a single numeric "features" vector per shot
that MLlib can learn from, and packages the transformation steps as reusable
"stages" so the model file can apply the EXACT same transforms to train and test
data — without leaking information from one into the other.

WHY THIS IS ITS OWN FILE
------------------------
Feature engineering is where most bugs live: a wrong column, a leaked target, or a
category the model has never seen. Isolating it lets you run THIS file alone, look
at the resulting vector, and convince yourself it's right before any modeling.

HOW TO USE IT
-------------
- Run it standalone (the __main__ block at the bottom): loads a small sample,
  builds the vector, and prints diagnostics so you can eyeball correctness.
- Import it from shot_model.py: call build_feature_stages() to get the list of
  stages, append your classifier, and wrap everything in ONE Pipeline.

ABOUT SPARK'S LAZINESS (worth internalizing)
--------------------------------------------
Spark is "lazy": transformations (select, withColumn, the indexers/encoders) only
record a PLAN — nothing computes. Work happens only when an ACTION (count, show,
collect, write, or fitting a model) forces it. That's why this script can describe
a big pipeline instantly and only does real compute at the .count()/.show() calls.
A side effect: each action re-runs the plan from scratch unless you .cache(), which
is why we cache the sampled DataFrame before hitting it with several actions.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler

# ============================================================================
# CONFIG — the only knobs you'll routinely touch
# ============================================================================

# The cleaned, feature-engineered Parquet produced in Phase 2 (partitioned by
# SEASON_1). Every analysis phase reads from here.
INPUT_PATH = "gs://pstat135-adam/processed/shots/"

# While developing, work on a 5% sample so jobs return in seconds. The vector's
# STRUCTURE is identical on a sample or the full data — only the row count differs —
# so this is safe for getting the pipeline correct.
#   >>> Set this to None for the final report run so the model sees all ~4.2M shots. <<<
SAMPLE_FRACTION = None

# One fixed seed everywhere = reproducible results (Lecture 18 reproducibility rule).
SEED = 42

# ============================================================================
# COLUMN ROLES — chosen deliberately; read the LEAKAGE warning before editing
# ============================================================================

# The thing we predict: 1 = made, 0 = missed (integer cast of the boolean SHOT_MADE).
LABEL_COL = "SHOT_MADE_INT"

# Numeric features used AS-IS. These are already numbers whose magnitude is
# meaningful (e.g. SHOT_DISTANCE is feet), so no transformation is needed.
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

# Categorical features (label-like text). Each must be indexed, then one-hot
# encoded, before a model can read it. ACTION_GROUP is created below from
# ACTION_TYPE; the other three already exist in the Parquet.
CATEGORICAL_COLS = [
    "BASIC_ZONE",    # 6 court zones (Restricted Area, Mid-Range, corners, etc.)
    "SHOT_TYPE",     # "2PT Field Goal" / "3PT Field Goal"
    "ZONE_RANGE",    # distance band
    "ACTION_GROUP",  # bucketed ACTION_TYPE (built in add_action_group)
]

# !!! LEAKAGE — NEVER ADD THESE AS FEATURES !!!
#   EXPECTED_PTS = SHOT_VALUE * SHOT_MADE_INT  -> equals 0 exactly when missed:
#                                                 it IS the answer in disguise.
#   EVENT_TYPE   = "Made Shot" / "Missed Shot" -> the target written as text.
#   SHOT_MADE    = boolean version of the target.
# If any of these reach the feature vector, ROC-AUC shoots toward 1.0 and the model
# is meaningless. A real shot-make model here should land around AUC 0.65-0.75 — so
# a near-perfect score is the warning sign that something leaked, not a triumph.


def add_action_group(df):
    """
    Collapse the many raw ACTION_TYPE values (e.g. "Driving Layup Shot",
    "Step Back Jump shot", "Turnaround Hook Shot") into 5 broad buckets.

    WHY: left raw, ACTION_TYPE has dozens of values, which one-hot encoding would
    blow up into dozens of mostly-empty columns — sparse noise that buries the
    signal. The signal we care about is HOW the shot was taken (at the rim vs a
    jumper vs a post move), and 5 buckets capture that cleanly.

    HOW: keyword matches on a lowercased copy of the column, checked top-to-bottom
    (first match wins, so order matters — "dunk"/"layup" are checked before the
    catch-all jump-shot keywords). The __main__ diagnostic prints the real values
    and the resulting bucket sizes so you can confirm "Other" isn't swallowing
    something important; if it is, add a keyword here.
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
    split only — which is what prevents test-set information from leaking in.

    THE THREE TRANSFORMS, AND WHY EACH EXISTS:

    1) StringIndexer — text category -> a number (e.g. "Mid-Range" -> 4.0).
       Models can't do arithmetic on strings, so categories must become numeric.
       handleInvalid="keep": if the test split contains a category never seen in
       training, it's parked in an extra bucket instead of crashing the job. (This
       is the concrete answer to "what would break this?" — an unseen category.)

    2) OneHotEncoder — that index -> a binary indicator vector. Necessary because
       otherwise the model reads index 5.0 as "greater than" 4.0, but those numbers
       are arbitrary labels (like jersey numbers — #23 isn't "more" than #22).
       dropLast=True (default) omits one column: if a row isn't any of the other
       categories it must be the dropped one, so keeping it would be redundant and
       make the feature matrix rank-deficient. Standard, expected practice.

    3) VectorAssembler — glue the numeric columns + every one-hot vector into ONE
       column literally named "features", because that's the column every Spark ML
       model reads from. handleInvalid="skip" drops any row carrying a null feature;
       the Phase-2 data is clean, so this is a guard rather than a workhorse — but
       the __main__ row-count print will reveal if it's silently dropping rows.
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

    # Order matters: indexers must run before the encoder, which runs before the
    # assembler. The list captures that dependency chain.
    return indexers + [encoder, assembler]


def main():
    # getOrCreate(): reuse the Spark session Dataproc already started for this job.
    spark = SparkSession.builder.appName("shot_feature_engineering").getOrCreate()
    # Quiet the noisy INFO logs so the diagnostics below are easy to find in output.
    spark.sparkContext.setLogLevel("WARN")

    # --- Load -----------------------------------------------------------------
    shots = spark.read.parquet(INPUT_PATH)
    if SAMPLE_FRACTION is not None:
        shots = shots.sample(fraction=SAMPLE_FRACTION, seed=SEED)

    # Build the bucketed action column now, then CACHE: we're about to trigger
    # several actions (counts, groupBys, a Pipeline fit, a transform). Without
    # caching, Spark would re-read Parquet and re-sample from scratch each time.
    shots = add_action_group(shots)
    shots.cache()

    print(f"\n[load] rows after sampling: {shots.count():,}")

    # --- Diagnostic: what ACTION_TYPE values actually exist? ------------------
    # (This is the groupBy worth running before trusting the buckets — baked in so
    #  it's checked against reality every run.)
    print("\n[diagnostic] top 30 raw ACTION_TYPE values:")
    shots.groupBy("ACTION_TYPE").count().orderBy(F.desc("count")).show(30, truncate=False)

    # --- Verify the buckets ---------------------------------------------------
    # If "Other" is large, an important action type is unmatched -> add a keyword
    # in add_action_group and rerun.
    print("[diagnostic] ACTION_GROUP bucket sizes:")
    shots.groupBy("ACTION_GROUP").count().orderBy(F.desc("count")).show(truncate=False)

    # --- Label balance --------------------------------------------------------
    # The base rate of makes. If a dumb model guessed "miss" every time, it would be
    # right (1 - make_rate) of the time — that's the accuracy floor any real model
    # must clear, and it's why accuracy alone is a weak metric (we'll lean on AUC/F1).
    print("[diagnostic] label balance (SHOT_MADE_INT):")
    shots.groupBy(LABEL_COL).count().orderBy(LABEL_COL).show()

    # --- Build + apply the feature stages -------------------------------------
    stages = build_feature_stages()
    # Standalone check only: fit a Pipeline of JUST the transformers (no model yet)
    # so we can inspect the "features" column it produces.
    feature_pipeline = Pipeline(stages=stages).fit(shots)
    transformed = feature_pipeline.transform(shots)

    # --- Verify the vector ----------------------------------------------------
    # Pull one row's vector and read its length. Sanity math:
    #   length = (# numeric features) + sum over categoricals of one-hot width,
    # where each one-hot width is roughly (distinct categories - 1) because of
    # dropLast (with small adjustments from handleInvalid="keep"). The printed
    # length below is the authoritative number; the per-column counts let you
    # reason about where it came from.
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

    # Compare this to the [load] count above: if it's lower, the assembler's
    # handleInvalid="skip" dropped rows with null features — investigate before
    # modeling rather than letting it pass silently.
    print(f"[done] rows available for modeling: {transformed.count():,}")

    spark.stop()


if __name__ == "__main__":
    main()