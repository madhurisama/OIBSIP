import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("Mall_Customers.csv")

print("=" * 50)
print("CUSTOMER SEGMENTATION DATASET")
print("=" * 50)
print(f"Shape   : {df.shape}")
print(f"Columns : {list(df.columns)}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nData Types:\n{df.dtypes}")



print("\n" + "=" * 50)
print("DATA CLEANING")
print("=" * 50)
print(f"Missing values:\n{df.isnull().sum()}")
print(f"Duplicates: {df.duplicated().sum()}")

df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

df.columns = ["CustomerID", "Gender", "Age", "Annual_Income", "Spending_Score"]


df["Gender_Encoded"] = df["Gender"].map({"Male": 0, "Female": 1})

print(f"\n✔ Columns renamed")
print(f"✔ Gender encoded (Male=0, Female=1)")
print(f"✔ Clean shape: {df.shape}")



print("\n" + "=" * 50)
print("DESCRIPTIVE STATISTICS")
print("=" * 50)
print(df.describe())
print(f"\nGender Distribution:\n{df['Gender'].value_counts()}")



features = df[["Annual_Income", "Spending_Score"]]
scaler   = StandardScaler()
features_scaled = scaler.fit_transform(features)

inertia    = []
silhouette = []
K_range    = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(features_scaled)
    inertia.append(km.inertia_)
    silhouette.append(silhouette_score(features_scaled, km.labels_))

print("\n── Silhouette Scores ──")
for k, s in zip(K_range, silhouette):
    print(f"  K={k} : {s:.4f}")

best_k = K_range[silhouette.index(max(silhouette))]
print(f"\n✔ Best K by Silhouette Score: {best_k}")



kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(features_scaled)


cluster_summary = df.groupby("Cluster").agg(
    Count         = ("CustomerID", "count"),
    Avg_Age       = ("Age", "mean"),
    Avg_Income    = ("Annual_Income", "mean"),
    Avg_Spending  = ("Spending_Score", "mean"),
    Female_Pct    = ("Gender_Encoded", "mean")
).round(2)
cluster_summary["Female_Pct"] = (cluster_summary["Female_Pct"] * 100).round(1)

print("\n── Cluster Summary ──")
print(cluster_summary)


cluster_labels = {
    df.groupby("Cluster")["Spending_Score"].mean().idxmax(): "High Spenders 💰",
    df.groupby("Cluster")["Spending_Score"].mean().idxmin(): "Low Spenders 💤",
    df.groupby("Cluster")["Annual_Income"].mean().idxmax(): "High Income 👑",
    df.groupby("Cluster")["Annual_Income"].mean().idxmin(): "Low Income 📉",
}
df["Cluster_Label"] = df["Cluster"].map(
    lambda x: cluster_labels.get(x, f"Cluster {x}")
)



sns.set_theme(style="whitegrid")
CLUSTER_PALETTE = ["#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6"]

fig = plt.figure(figsize=(22, 26))
fig.suptitle("Customer Segmentation Dashboard — Madhuri",
             fontsize=22, fontweight="bold", y=1.01)



ax1 = fig.add_subplot(4, 3, 1)
ax1.plot(K_range, inertia, marker="o", color="#e74c3c", linewidth=2)
ax1.set_title("Elbow Method — Optimal K", fontsize=12, fontweight="bold")
ax1.set_xlabel("Number of Clusters (K)")
ax1.set_ylabel("Inertia")
ax1.axvline(5, color="gray", linestyle="--", label="K=5")
ax1.legend()

ax2 = fig.add_subplot(4, 3, 2)
ax2.plot(K_range, silhouette, marker="s", color="#2ecc71", linewidth=2)
ax2.set_title("Silhouette Score vs K", fontsize=12, fontweight="bold")
ax2.set_xlabel("Number of Clusters (K)")
ax2.set_ylabel("Silhouette Score")
ax2.axvline(best_k, color="red", linestyle="--", label=f"Best K={best_k}")
ax2.legend()


ax3 = fig.add_subplot(4, 3, 3)
for i in range(5):
    mask = df["Cluster"] == i
    ax3.scatter(df[mask]["Annual_Income"],
                df[mask]["Spending_Score"],
                c=CLUSTER_PALETTE[i], label=f"Cluster {i}",
                s=80, alpha=0.8, edgecolors="white")
centers = scaler.inverse_transform(kmeans.cluster_centers_)
ax3.scatter(centers[:, 0], centers[:, 1],
            c="black", marker="X", s=200, label="Centroids", zorder=5)
ax3.set_title("Customer Segments", fontsize=12, fontweight="bold")
ax3.set_xlabel("Annual Income (k$)")
ax3.set_ylabel("Spending Score (1-100)")
ax3.legend(fontsize=8)



ax4 = fig.add_subplot(4, 3, 4)
cluster_counts = df["Cluster"].value_counts().sort_index()
ax4.bar([f"Cluster {i}" for i in cluster_counts.index],
        cluster_counts.values,
        color=CLUSTER_PALETTE, edgecolor="white")
ax4.set_title("Customers per Cluster", fontsize=12, fontweight="bold")
ax4.set_ylabel("Count")
for i, v in enumerate(cluster_counts.values):
    ax4.text(i, v + 0.5, str(v), ha="center", fontweight="bold")



ax5 = fig.add_subplot(4, 3, 5)
avg_income = df.groupby("Cluster")["Annual_Income"].mean()
ax5.bar([f"Cluster {i}" for i in avg_income.index],
        avg_income.values, color=CLUSTER_PALETTE, edgecolor="white")
ax5.set_title("Avg Annual Income by Cluster", fontsize=12, fontweight="bold")
ax5.set_ylabel("Annual Income (k$)")
for i, v in enumerate(avg_income.values):
    ax5.text(i, v + 0.5, f"{v:.1f}k", ha="center", fontweight="bold", fontsize=9)



ax6 = fig.add_subplot(4, 3, 6)
avg_spending = df.groupby("Cluster")["Spending_Score"].mean()
ax6.bar([f"Cluster {i}" for i in avg_spending.index],
        avg_spending.values, color=CLUSTER_PALETTE, edgecolor="white")
ax6.set_title("Avg Spending Score by Cluster", fontsize=12, fontweight="bold")
ax6.set_ylabel("Spending Score")
for i, v in enumerate(avg_spending.values):
    ax6.text(i, v + 0.5, f"{v:.1f}", ha="center", fontweight="bold", fontsize=9)



ax7 = fig.add_subplot(4, 3, 7)
for i in range(5):
    ax7.hist(df[df["Cluster"] == i]["Age"], bins=15,
             alpha=0.6, color=CLUSTER_PALETTE[i], label=f"Cluster {i}")
ax7.set_title("Age Distribution by Cluster", fontsize=12, fontweight="bold")
ax7.set_xlabel("Age")
ax7.set_ylabel("Count")
ax7.legend(fontsize=8)



ax8 = fig.add_subplot(4, 3, 8)
gender_cluster = df.groupby(["Cluster", "Gender"]).size().unstack()
gender_cluster.plot(kind="bar", ax=ax8,
                    color=["#3498db", "#e91e8c"],
                    edgecolor="white")
ax8.set_title("Gender Distribution by Cluster", fontsize=12, fontweight="bold")
ax8.set_xlabel("Cluster")
ax8.set_ylabel("Count")
ax8.set_xticklabels([f"C{i}" for i in range(5)], rotation=0)
ax8.legend(["Male", "Female"])



ax9 = fig.add_subplot(4, 3, 9)
for i in range(5):
    mask = df["Cluster"] == i
    ax9.scatter(df[mask]["Age"], df[mask]["Annual_Income"],
                c=CLUSTER_PALETTE[i], label=f"Cluster {i}",
                s=60, alpha=0.8, edgecolors="white")
ax9.set_title("Age vs Annual Income by Cluster", fontsize=12, fontweight="bold")
ax9.set_xlabel("Age")
ax9.set_ylabel("Annual Income (k$)")
ax9.legend(fontsize=8)



ax10 = fig.add_subplot(4, 3, 10)
for i in range(5):
    mask = df["Cluster"] == i
    ax10.scatter(df[mask]["Age"], df[mask]["Spending_Score"],
                 c=CLUSTER_PALETTE[i], label=f"Cluster {i}",
                 s=60, alpha=0.8, edgecolors="white")
ax10.set_title("Age vs Spending Score by Cluster", fontsize=12, fontweight="bold")
ax10.set_xlabel("Age")
ax10.set_ylabel("Spending Score")
ax10.legend(fontsize=8)



ax11 = fig.add_subplot(4, 3, 11)
df.boxplot(column="Annual_Income", by="Cluster",
           patch_artist=True, ax=ax11)
ax11.set_title("Income Distribution by Cluster", fontsize=12, fontweight="bold")
ax11.set_xlabel("Cluster")
ax11.set_ylabel("Annual Income (k$)")
plt.suptitle("")



ax12 = fig.add_subplot(4, 3, 12)
summary_heat = cluster_summary[["Avg_Age","Avg_Income","Avg_Spending","Female_Pct"]]
sns.heatmap(summary_heat, annot=True, fmt=".1f",
            cmap="YlOrRd", linewidths=0.5, ax=ax12)
ax12.set_title("Cluster Profile Heatmap", fontsize=12, fontweight="bold")


plt.tight_layout()
plt.savefig("Madhuri_Task3_CustomerSegmentation.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Dashboard saved!")


print("\n" + "=" * 50)
print("         KEY FINDINGS")
print("=" * 50)
print(cluster_summary.to_string())
print(f"\nOptimal Clusters (Silhouette): {best_k}")
print(f"Total Customers Segmented    : {len(df)}")
print("\n── Segment Insights ──")
print("Cluster 0 — Low Income, Low Spending  : Need budget-friendly offers")
print("Cluster 1 — High Income, Low Spending : Target with premium campaigns")
print("Cluster 2 — Low Income, High Spending : Loyal but risky — retain them")
print("Cluster 3 — High Income, High Spending: VIP customers — reward them!")
print("Cluster 4 — Middle Income, Mid Spending: Largest segment — standard offers")