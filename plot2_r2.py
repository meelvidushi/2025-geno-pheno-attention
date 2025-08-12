import pandas as pd
import matplotlib.pyplot as plt

# Load metrics
df = pd.read_csv("models/20250716_215606/lightning_logs/version_1507291/metrics.csv")

# Filter for R² metrics only
r2_df = df[df["metric"].str.contains("r2")].copy()

# Clean column names for clarity
r2_df["metric_clean"] = (
    r2_df["metric"]
    .str.replace("val_", "Val ", regex=False)
    .str.replace("test_", "Test ", regex=False)
    .str.replace("_30C", "30C", regex=False)
    .str.replace("_r2", "R²", regex=False)
    .str.replace("r2", "R²", regex=False)
)

# Plot
plt.figure(figsize=(10, 5))
plt.bar(r2_df["metric_clean"], r2_df["value"], color="#4da6ff", edgecolor="black")

plt.title("📈 R² Scores for 30°C Phenotype (Arcadia Transformer)")
plt.ylabel("R²")
plt.ylim(0.55, 0.65)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("r2_better_plot.png")
plt.show()
