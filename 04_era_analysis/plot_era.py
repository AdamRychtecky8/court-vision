"""
plot_era.py — Figure 2: NBA three-point attempt rate by season (Act 2 / era analysis)
=====================================================================================
Runs LOCALLY on your machine — no Dataproc, no Spark. The era table is only 21 rows
(one per season), so there is nothing "big" left to do here; the heavy aggregation
already happened in Phase 2. This script just renders those already-computed numbers
into the figure that anchors the report's era story.

INPUT  (download once from GCS, from the court-vision/ root):
    gcloud storage cp `
      "gs://pstat135-adam/processed/tables/three_pt_rate_by_season/part-*.csv" `
      04_era_analysis/three_pt_rate_by_season.csv

OUTPUT:
    report/fig2_three_pt_rate_by_season.png

DEPENDENCIES:  pandas + matplotlib.  If missing:  pip install pandas matplotlib

NOTE ON THE DATA: three_pt_rate_pct is ALREADY a percentage (e.g. 18.6), so no rate
math is needed here. The rate rises straight through the 2012 lockout and 2020 COVID
seasons — those years only dip in total ATTEMPTS (fewer games), not in the rate.
"""

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")            # render to a file rather than a screen window — safe everywhere
import matplotlib.pyplot as plt  # must be imported AFTER setting the backend above

# Locate files relative to THIS script so it works no matter what directory you run from.
HERE = Path(__file__).resolve().parent
CSV_IN = HERE / "three_pt_rate_by_season.csv"
PNG_OUT = HERE.parent / "report" / "fig2_three_pt_rate_by_season.png"

# --- Load the era table ---------------------------------------------------
# Columns: SEASON_1, total_attempts, three_pt_attempts, three_pt_rate_pct, overall_fg_pct
df = pd.read_csv(CSV_IN).sort_values("SEASON_1").reset_index(drop=True)
seasons = df["SEASON_1"]
rate = df["three_pt_rate_pct"]

# --- Build the figure -----------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

# The rate line. Markers make each season a readable data point.
ax.plot(seasons, rate, marker="o", linewidth=2.2, markersize=5,
        color="#C8102E", zorder=3)   # NBA red

# Shade the 2015-2017 acceleration window — the inflection the report is built around.
ax.axvspan(2015, 2017, color="#1D428A", alpha=0.10, zorder=1)  # faint NBA blue
ax.text(2016, 16.0, "Acceleration\n2015–2017", ha="center", va="bottom",
        fontsize=9, color="#1D428A")

# Annotate the trigger near the 2016 point (Curry's unanimous MVP, 73-win Warriors).
y2016 = float(df.loc[df["SEASON_1"] == 2016, "three_pt_rate_pct"].iloc[0])
ax.annotate("Warriors dynasty;\nCurry unanimous MVP (2016)",
            xy=(2016, y2016), xytext=(2008.5, 35),
            fontsize=9, color="#333333",
            arrowprops=dict(arrowstyle="->", color="#888888"))

# Label the flat decade so the before/after contrast is obvious.
ax.text(2008.5, 20.0, "Flat era ~22%", fontsize=9, color="#666666", style="italic")

# --- Axis cosmetics -------------------------------------------------------
ax.set_title("The Three-Point Revolution: NBA 3PT Attempt Rate by Season (2004–2024)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Season (start year)")
ax.set_ylabel("Share of field-goal attempts from three (%)")
ax.set_xticks(range(2004, 2025, 2))
ax.set_ylim(15, 45)
ax.grid(True, axis="y", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
PNG_OUT.parent.mkdir(parents=True, exist_ok=True)   # create report/ if it doesn't exist
fig.savefig(PNG_OUT, dpi=200, bbox_inches="tight")
print(f"wrote {PNG_OUT}")

# --- Quick talking points for the writeup ---------------------------------
df["yoy_change"] = df["three_pt_rate_pct"].diff()
steepest = df.loc[df["yoy_change"].idxmax()]
print(f"2004 rate: {rate.iloc[0]:.1f}%  ->  2024 rate: {rate.iloc[-1]:.1f}%  "
      f"({rate.iloc[-1] - rate.iloc[0]:+.1f} points over 21 seasons)")
print(f"Steepest single-season jump: {int(steepest['SEASON_1'])-1}->{int(steepest['SEASON_1'])}"
      f" = +{steepest['yoy_change']:.1f} points")