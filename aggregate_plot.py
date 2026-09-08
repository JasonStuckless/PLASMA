import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO

csv_data = """chunk_duration_ms,pcr,por,tci,pli,baseline_count,matched_count,omitted_count
100,0.045148485707496445,0.9548515142925036,1.1787482034407373,0.7171838260683148,144.2,6.533333333333333,137.66666666666666
150,0.21173482724298984,0.7882651727570101,0.9963557397080486,0.5582729522896012,144.2,30.5,113.7
200,0.291510607825981,0.7084893921740191,1.0227447365449933,0.5054842583788066,144.2,42.1,102.1
250,0.4437491852954476,0.5562508147045525,0.9766055995913051,0.3987636574457151,144.2,64.2,80.0
300,0.5021802689251695,0.4978197310748305,1.0080871581559816,0.35533394485246145,144.2,72.56666666666666,71.63333333333334
350,0.5913584371678243,0.40864156283217573,0.9775450991354143,0.29424572029918156,144.2,85.63333333333334,58.56666666666667
400,0.6204214359852015,0.3795785640147985,1.0066240572752347,0.272762616892155,144.2,89.7,54.5
"""

# Read CSV into DataFrame
df = pd.read_csv(StringIO(csv_data))

# Melt for plotting
plot_df = df[["chunk_duration_ms", "pcr", "por", "pli"]].melt(
    id_vars="chunk_duration_ms",
    var_name="metric",
    value_name="value"
)

# Rename for legend
plot_df["metric"] = plot_df["metric"].replace({
    "pcr": "PCR",
    "por": "POR",
    "pli": "PLI"
})

sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

sns.lineplot(
    data=plot_df,
    x="chunk_duration_ms",
    y="value",
    hue="metric",
    style="metric",
    markers={
        "PCR": "o",   # circle
        "POR": "s",   # square
        "PLI": "^"    # triangle
    },
    dashes=False,  # keep solid lines (optional: set True if you want extra differentiation)
    linewidth=2.5,
    markersize=8
)

plt.title("PCR, POR, and PLI vs Chunk Duration", fontsize=16)
plt.xlabel("Chunk duration (ms)", fontsize=12)
plt.ylabel("Metric value", fontsize=12)
plt.xticks(df["chunk_duration_ms"])
plt.legend(title="Metric")
plt.tight_layout()
plt.show()