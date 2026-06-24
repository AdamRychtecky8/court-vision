"""
Figure 2: NBA three-point attempt rate by season (Act 2 / era analysis)
"""

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")            
import matplotlib.pyplot as plt  

HERE = Path(__file__).resolve().parent
CSV_IN = HERE / "three_pt_rate_by_season.csv"
PNG_OUT = HERE.parent / "report" / "fig2_three_pt_rate_by_season.png"

# Columns: SEASON_1, total_attempts, three_pt_attempts, three_pt_rate_pct, overall_fg_pct
df = pd.read_csv(CSV_IN).sort_values("SEASON_1").reset_index(drop=True)
seasons = df["SEASON_1"]
rate = df["three_pt_rate_pct"]

fig, ax = plt.subplots(figsize=(10, 6))

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
PNG_OUT.parent.mkdir(parents=True, exist_ok=True)   
fig.savefig(PNG_OUT, dpi=200, bbox_inches="tight")
print(f"wrote {PNG_OUT}")

df["yoy_change"] = df["three_pt_rate_pct"].diff()
steepest = df.loc[df["yoy_change"].idxmax()]
print(f"2004 rate: {rate.iloc[0]:.1f}%  ->  2024 rate: {rate.iloc[-1]:.1f}%  "
      f"({rate.iloc[-1] - rate.iloc[0]:+.1f} points over 21 seasons)")
print(f"Steepest single-season jump: {int(steepest['SEASON_1'])-1}->{int(steepest['SEASON_1'])}"
      f" = +{steepest['yoy_change']:.1f} points")