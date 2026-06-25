from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

HERE = Path(__file__).resolve().parent
CSV_IN = HERE.parent / "report" / "tables" / "lasso_path.csv"
PNG_OUT = HERE.parent / "report" / "fig3_lasso_path.png"

df = pd.read_csv(CSV_IN)

# Count non-zero coefficients per regParam
counts = df[df['coefficient'] != 0.0].groupby('regParam')['feature'].count().reset_index()
counts.columns = ['regParam', 'n_features']
counts = counts.sort_values('regParam', ascending=False)

# For the line plot: pick the top features by max absolute coefficient across all regParams
pivoted = df.pivot(index='regParam', columns='feature', values='coefficient').fillna(0)
pivoted = pivoted.sort_index(ascending=False)

# Identify top 8 features by max absolute value (excluding near-zero everywhere)
max_abs = pivoted.abs().max(axis=0)
top_features = max_abs.nlargest(8).index.tolist()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: coefficient paths for top features
ax1 = axes[0]
colors = plt.cm.tab10(np.linspace(0, 1, len(top_features)))
for i, feat in enumerate(top_features):
    label = feat.replace('_ohe_', ': ').replace('ACTION_GROUP: ', '').replace('BASIC_ZONE: ', '').replace('ZONE_RANGE: ', '').replace('SHOT_DISTANCE', 'Distance').replace('IS_CLUTCH', 'Clutch').replace('QUARTER', 'Quarter')
    ax1.plot(pivoted.index, pivoted[feat], marker='o', markersize=4, label=label, color=colors[i])

ax1.set_xscale('log')
ax1.invert_xaxis()
ax1.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax1.set_xlabel('Regularization Penalty (regParam, log scale)', fontsize=10)
ax1.set_ylabel('LASSO Coefficient', fontsize=10)
ax1.set_title('LASSO Regularization Path\n(Top 8 Features by Magnitude)', fontsize=11)
ax1.legend(fontsize=7, loc='upper left')
ax1.grid(True, alpha=0.3)

# Right: number of active features vs regParam
ax2 = axes[1]
ax2.plot(counts['regParam'], counts['n_features'], marker='s', color='steelblue', linewidth=2)
ax2.set_xscale('log')
ax2.invert_xaxis()
ax2.set_xlabel('Regularization Penalty (regParam, log scale)', fontsize=10)
ax2.set_ylabel('Active Features (nonzero coeff)', fontsize=10)
ax2.set_title('Feature Sparsity vs. Penalty\n(LASSO Path Summary)', fontsize=11)
ax2.grid(True, alpha=0.3)
for _, row in counts.iterrows():
    ax2.annotate(str(int(row['n_features'])), (row['regParam'], row['n_features']),
                 textcoords='offset points', xytext=(0, 8), ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(PNG_OUT, dpi=150, bbox_inches='tight')
print(f"Saved {PNG_OUT}")
