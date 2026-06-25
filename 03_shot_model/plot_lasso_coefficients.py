"""
LASSO coefficient bar chart

DESIGN NOTES:
- The "Back Court Shot" residual (~ -1.34) is EXCLUDED so it doesn't stretch the axis
  and squash the meaningful bars.
- Bars are colored by category (Action / Location / Context).
"""

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CSV_IN = HERE.parent / "report" / "tables" / "lasso_coefficients.csv" 
PNG_OUT = HERE.parent / "report" / "fig4_lasso_coefficients.png"

EXCLUDE = ["Back Court"]  
THRESHOLD = 0.02           # drop negligible coefficients to reduce clutter

CATEGORY_COLORS = {
    "Action":   "#C8102E",  
    "Location": "#1D428A",  
    "Context":  "#9A9A9A",  # muted; these sit near zero
}

CONTEXT_FEATURES = {"QUARTER", "IS_CLUTCH", "SECS_REMAINING"}

def categorize(name):
    if name.startswith("ACTION_GROUP_ohe_"):
        return "Action"
    if name in CONTEXT_FEATURES:
        return "Context"
    return "Location"

def clean_feat(name):
    s = name
    s = s.replace("ACTION_GROUP_ohe_", "Action: ")
    s = s.replace("BASIC_ZONE_ohe_", "Zone: ")
    s = s.replace("ZONE_RANGE_ohe_", "Range: ")
    s = s.replace("SHOT_TYPE_ohe_", "Type: ")
    s = s.replace("IS_CLUTCH", "Clutch Situation (Q4/OT, ≤120 s)")
    s = s.replace("SHOT_DISTANCE", "Shot Distance (ft)")
    s = s.replace("QUARTER", "Quarter")
    s = s.replace("SECS_REMAINING", "Seconds Remaining")
    s = s.replace("LOC_Y", "Court Y Coordinate")
    s = s.replace("LOC_X", "Court X Coordinate")
    s = s.replace("SHOT_VALUE", "Shot Value (2 or 3)")
    s = s.replace("IS_3PT", "Is Three-Point Attempt")
    return s

df = pd.read_csv(CSV_IN)
df = df[~df["feature"].str.contains("|".join(EXCLUDE), case=False)].copy()

dropped = df[df["coefficient"].abs() < THRESHOLD].copy()
print(f"Dropped {len(dropped)} feature(s) below |coefficient| < {THRESHOLD}:")
for _, row in dropped.sort_values("coefficient").iterrows():
    print(f"  {row['coefficient']:+.4f}  {row['feature']}")

df = df[df["coefficient"].abs() >= THRESHOLD].copy()
df["category"] = df["feature"].apply(categorize)
df["label"] = df["feature"].apply(clean_feat)
df = df.sort_values("coefficient").reset_index(drop=True)  # most negative bottom -> most positive top

fig, ax = plt.subplots(figsize=(9, 7))
colors = df["category"].map(CATEGORY_COLORS)
ax.barh(df["label"], df["coefficient"], color=colors, zorder=3)
ax.axvline(0, color="black", linewidth=0.8, zorder=2)

ax.set_xlabel("LASSO coefficient  (negative = lower make probability  →  positive = higher)")
ax.set_title("What Predicts a Made Shot: LASSO Coefficients (regParam = 0.001)",
             fontsize=12, fontweight="bold")
ax.grid(True, axis="x", alpha=0.3, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in CATEGORY_COLORS.values()]
ax.legend(handles, CATEGORY_COLORS.keys(), title="Feature type", loc="lower right")

fig.tight_layout()
PNG_OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(PNG_OUT, dpi=200, bbox_inches="tight")
print(f"wrote {PNG_OUT}")