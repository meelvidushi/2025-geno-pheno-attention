import pandas as pd
import matplotlib.pyplot as plt

# Load your CSV
df = pd.read_csv("models/20250716_215606/lightning_logs/version_1507291/metrics.csv")

# Filter for R² metrics only
r2_df = df[df["metric"].str.contains("r2") & ~df["metric"].str.contains("loss")]

# Bar plot
plt.figure(figsize=(8, 5))
plt.bar(r2_df["metric"], r2_df["value"], color="skyblue")
plt.title("R² Scores for 30°C Phenotype")
plt.ylabel("R²")
plt.xticks(rotation=30)
plt.ylim(0.5, 0.65)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("r2_plot.png")
plt.show()
