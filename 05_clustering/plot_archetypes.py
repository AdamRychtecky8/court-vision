"""
plot_archetypes.py — Act 3 figures: the archetype map (Fig 5) and the era shift (Fig 6)
=======================================================================================
Runs LOCALLY (no Spark). Reads the two CSVs saved by archetypes.py.

INPUTS (download once from the court-vision/ root):
    gcloud storage cp `
      "gs://pstat135-adam/processed/tables/player_clusters/part-*.csv" `
      05_clustering/player_clusters.csv
    gcloud storage cp `
      "gs://pstat135-adam/processed/tables/cluster_by_era/part-*.csv" `
      05_clustering/cluster_by_era.csv

OUTPUTS:
    report/fig5_archetype_map.png
    report/fig6_archetype_era_shift.png

DEPENDENCIES: pandas + matplotlib   (pip install pandas matplotlib)

CLUSTER NUMBERING: the cluster IDs come from K-Means with seed=42 on the full data, so
they are reproducible. The names below are read off the Step-5 profile table — verify them
against that table and edit if you reran with a different k or seed.
"""

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CLUSTERS_CSV = HERE / "player_clusters.csv"
ERA_CSV = HERE / "cluster_by_era.csv"
FIG5_OUT = HERE.parent / "report" / "fig5_archetype_map.png"
FIG6_OUT = HERE.parent / "report" / "fig6_archetype_era_shift.png"

# Cluster ID -> human name + a stable color. Confirm against the profile table.
CLUSTER_NAMES = {
    0: "Balanced Wing",
    1: "Rim-Running Big",
    2: "Floor Spacer",
    3: "Perimeter Creator",
    4: "Mid-Range Scorer",
}
CLUSTER_COLORS = {
    0: "#1D428A",  # blue
    1: "#C8102E",  # red
    2: "#2E9E5B",  # green
    3: "#F2A900",  # gold
    4: "#7D3C98",  # purple
}

# ===========================================================================
# Figure 5 — archetype map: every player-season placed on the two PCA axes
# ===========================================================================
df = pd.read_csv(CLUSTERS_CSV)

fig, ax = plt.subplots(figsize=(10, 8))
for cid, name in CLUSTER_NAMES.items():
    sub = df[df["cluster"] == cid]
    ax.scatter(sub["PC1"], sub["PC2"], s=10, alpha=0.30,
               color=CLUSTER_COLORS[cid], label=f"{name} (n={len(sub)})")
    # Plot the centroid with a labeled marker so each region of the map is named.
    cx, cy = sub["PC1"].mean(), sub["PC2"].mean()
    ax.scatter(cx, cy, s=180, color=CLUSTER_COLORS[cid],
               edgecolor="black", linewidth=1.4, zorder=5)
    ax.annotate(name, (cx, cy), fontsize=10, fontweight="bold",
                ha="center", va="center", color="white", zorder=6)

# Axis labels spell out what the PCA axes MEAN (from the loadings), so the map is readable.
ax.set_xlabel("PC1 (51.5%):   interior  ←——————→  perimeter", fontsize=11)
ax.set_ylabel("PC2 (23.0%):   mid-range  ←——————→  rim / paint", fontsize=11)
ax.set_title("Player Archetypes by Shot-Location Profile (2004–2024)",
             fontsize=13, fontweight="bold")
ax.axhline(0, color="#cccccc", linewidth=0.8, zorder=0)
ax.axvline(0, color="#cccccc", linewidth=0.8, zorder=0)
ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
FIG5_OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG5_OUT, dpi=200, bbox_inches="tight")
print(f"wrote {FIG5_OUT}")

# ===========================================================================
# Figure 6 — era shift: each archetype's share of player-seasons, pre vs post 2015
# ===========================================================================
era = pd.read_csv(ERA_CSV).sort_values("cluster").reset_index(drop=True)
era["name"] = era["cluster"].map(CLUSTER_NAMES)
# Order bars by how much each archetype grew/shrank, so the story reads left-to-right.
era = era.sort_values("delta_pct", ascending=False).reset_index(drop=True)

x = range(len(era))
width = 0.38
fig2, ax2 = plt.subplots(figsize=(11, 6))
ax2.bar([i - width / 2 for i in x], era["pre_pct"], width,
        label="Pre-2015", color="#A9A9A9")
ax2.bar([i + width / 2 for i in x], era["post_pct"], width,
        label="Post-2015", color="#1D428A")

# Annotate each pair with the change in percentage points.
for i, row in era.iterrows():
    ax2.text(i, max(row["pre_pct"], row["post_pct"]) + 0.8,
             f"{row['delta_pct']:+.1f}", ha="center", fontsize=9,
             color=("#2E9E5B" if row["delta_pct"] >= 0 else "#C8102E"), fontweight="bold")

ax2.set_xticks(list(x))
ax2.set_xticklabels(era["name"], rotation=15, ha="right")
ax2.set_ylabel("Share of era's player-seasons (%)")
ax2.set_title("How Player Roles Reorganized After 2015", fontsize=13, fontweight="bold")
ax2.legend(fontsize=10)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.grid(True, axis="y", alpha=0.3)

fig2.tight_layout()
fig2.savefig(FIG6_OUT, dpi=200, bbox_inches="tight")
print(f"wrote {FIG6_OUT}")