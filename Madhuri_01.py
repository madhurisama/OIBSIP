import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("netflix_titles.csv")

print("=" * 50)
print("ORIGINAL DATASET")
print("=" * 50)
print(f"Shape        : {df.shape}")
print(f"Columns      : {list(df.columns)}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nData Types:\n{df.dtypes}")



print("\n" + "=" * 50)
print("MISSING VALUES — BEFORE CLEANING")
print("=" * 50)
missing_before = df.isnull().sum()
missing_pct_before = (df.isnull().sum() / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    "Missing Count" : missing_before,
    "Missing %" : missing_pct_before
})
print(missing_df[missing_df["Missing Count"] > 0])

duplicates_before = df.duplicated().sum()
print(f"\nDuplicate rows : {duplicates_before}")



print("\n" + "=" * 50)
print("CLEANING STEPS")
print("=" * 50)

df_clean = df.copy()


df_clean.drop_duplicates(inplace=True)
print(f"✔ Duplicates removed     : {duplicates_before - df_clean.duplicated().sum()}")

df_clean["director"].fillna("Unknown", inplace=True)
print(f"✔ 'director' filled with 'Unknown'")


df_clean["cast"].fillna("Unknown", inplace=True)
print(f"✔ 'cast' filled with 'Unknown'")


df_clean["country"].fillna(df_clean["country"].mode()[0], inplace=True)
print(f"✔ 'country' filled with mode: {df['country'].mode()[0]}")


df_clean["date_added"].fillna(df_clean["date_added"].mode()[0], inplace=True)
print(f"✔ 'date_added' filled with mode")


df_clean["rating"].fillna(df_clean["rating"].mode()[0], inplace=True)
print(f"✔ 'rating' filled with mode: {df['rating'].mode()[0]}")

df_clean["duration"].fillna(df_clean["duration"].mode()[0], inplace=True)
print(f"✔ 'duration' filled with mode")

df_clean["type"]    = df_clean["type"].str.strip().str.title()
df_clean["country"] = df_clean["country"].str.strip().str.title()
df_clean["rating"]  = df_clean["rating"].str.strip().str.upper()
print(f"✔ Text columns standardized (strip + title/upper case)")


df_clean["date_added"] = pd.to_datetime(
    df_clean["date_added"].str.strip(), format="%B %d, %Y", errors="coerce"
)
df_clean["year_added"]  = df_clean["date_added"].dt.year
df_clean["month_added"] = df_clean["date_added"].dt.month
print(f"✔ 'date_added' converted to datetime, extracted year & month")


df_clean["duration_value"] = df_clean["duration"].str.extract(r"(\d+)").astype(float)
df_clean["duration_unit"]  = df_clean["duration"].str.extract(r"([a-zA-Z]+)")
print(f"✔ 'duration' split into 'duration_value' and 'duration_unit'")


Q1 = df_clean["release_year"].quantile(0.25)
Q3 = df_clean["release_year"].quantile(0.75)
IQR = Q3 - Q1
outliers = df_clean[
    (df_clean["release_year"] < Q1 - 1.5 * IQR) |
    (df_clean["release_year"] > Q3 + 1.5 * IQR)
]
print(f"✔ Outliers in 'release_year': {len(outliers)} rows detected")


print("\n" + "=" * 50)
print("MISSING VALUES — AFTER CLEANING")
print("=" * 50)
missing_after = df_clean.isnull().sum()
print(missing_after[missing_after > 0] if missing_after.sum() > 0 else "✅ No missing values!")
print(f"\nFinal Shape  : {df_clean.shape}")
print(f"Duplicates   : {df_clean.duplicated().sum()}")


# ══════════════════════════════════════════════
# 5. SAVE CLEANED DATA
# ══════════════════════════════════════════════
df_clean.to_csv("netflix_cleaned.csv", index=False)
print("\n✅ Cleaned dataset saved as 'netflix_cleaned.csv'")



sns.set_theme(style="whitegrid")
fig = plt.figure(figsize=(20, 22))
fig.suptitle("Netflix Data Cleaning — Analysis Dashboard",
             fontsize=22, fontweight="bold", y=1.01)


ax1 = fig.add_subplot(4, 3, 1)
missing_plot = missing_df[missing_df["Missing Count"] > 0]
ax1.barh(missing_plot.index, missing_plot["Missing %"],
         color="#e74c3c", edgecolor="white")
ax1.set_title("Missing Values % — Before Cleaning", fontsize=12, fontweight="bold")
ax1.set_xlabel("Missing %")
for i, v in enumerate(missing_plot["Missing %"]):
    ax1.text(v + 0.2, i, f"{v}%", va="center", fontsize=9)



ax2 = fig.add_subplot(4, 3, 2)
missing_after_plot = df_clean.isnull().sum()
missing_after_plot = missing_after_plot[missing_after_plot > 0]
if len(missing_after_plot) == 0:
    ax2.text(0.5, 0.5, "✅ No Missing\nValues!",
             ha="center", va="center", fontsize=16,
             color="#2ecc71", fontweight="bold",
             transform=ax2.transAxes)
    ax2.set_title("Missing Values — After Cleaning", fontsize=12, fontweight="bold")
else:
    ax2.barh(missing_after_plot.index, missing_after_plot.values,
             color="#2ecc71", edgecolor="white")
    ax2.set_title("Missing Values — After Cleaning", fontsize=12, fontweight="bold")
ax2.axis("off") if len(missing_after_plot) == 0 else None



ax3 = fig.add_subplot(4, 3, 3)
type_counts = df_clean["type"].value_counts()
ax3.pie(type_counts.values, labels=type_counts.index,
        autopct="%1.1f%%", colors=["#3498db", "#e74c3c"],
        startangle=90)
ax3.set_title("Movies vs TV Shows", fontsize=12, fontweight="bold")


ax4 = fig.add_subplot(4, 3, 4)
top_countries = df_clean["country"].value_counts().head(10)
ax4.barh(top_countries.index[::-1], top_countries.values[::-1],
         color="#9b59b6", edgecolor="white")
ax4.set_title("Top 10 Countries by Content", fontsize=12, fontweight="bold")
ax4.set_xlabel("Count")



ax5 = fig.add_subplot(4, 3, 5)
year_counts = df_clean["year_added"].value_counts().sort_index()
ax5.plot(year_counts.index, year_counts.values,
         marker="o", color="#2ecc71", linewidth=2)
ax5.fill_between(year_counts.index, year_counts.values,
                 alpha=0.3, color="#2ecc71")
ax5.set_title("Content Added per Year", fontsize=12, fontweight="bold")
ax5.set_xlabel("Year")
ax5.set_ylabel("Count")



ax6 = fig.add_subplot(4, 3, 6)
rating_counts = df_clean["rating"].value_counts().head(10)
ax6.bar(rating_counts.index, rating_counts.values,
        color="#e67e22", edgecolor="white")
ax6.set_title("Content by Rating", fontsize=12, fontweight="bold")
ax6.set_ylabel("Count")
ax6.set_xticklabels(rating_counts.index, rotation=45, ha="right")



ax7 = fig.add_subplot(4, 3, 7)
ax7.hist(df_clean["release_year"], bins=40,
         color="#3498db", edgecolor="white")
ax7.set_title("Release Year Distribution", fontsize=12, fontweight="bold")
ax7.set_xlabel("Release Year")
ax7.set_ylabel("Count")



ax8 = fig.add_subplot(4, 3, 8)
month_names = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]
month_counts = df_clean["month_added"].value_counts().sort_index()
ax8.bar([month_names[m-1] for m in month_counts.index],
        month_counts.values, color="#1abc9c", edgecolor="white")
ax8.set_title("Content Added by Month", fontsize=12, fontweight="bold")
ax8.set_ylabel("Count")
ax8.set_xticklabels([month_names[m-1] for m in month_counts.index],
                    rotation=45, ha="right")



ax9 = fig.add_subplot(4, 3, 9)
top_directors = (df_clean[df_clean["director"] != "Unknown"]
                 ["director"].value_counts().head(10))
ax9.barh(top_directors.index[::-1], top_directors.values[::-1],
         color="#e91e8c", edgecolor="white")
ax9.set_title("Top 10 Directors", fontsize=12, fontweight="bold")
ax9.set_xlabel("Count")



ax10 = fig.add_subplot(4, 3, 10)
movies = df_clean[df_clean["duration_unit"] == "min"]["duration_value"]
ax10.hist(movies, bins=30, color="#f39c12", edgecolor="white")
ax10.axvline(movies.mean(), color="red", linestyle="--",
             label=f"Mean: {movies.mean():.0f} min")
ax10.set_title("Movie Duration Distribution", fontsize=12, fontweight="bold")
ax10.set_xlabel("Duration (minutes)")
ax10.set_ylabel("Count")
ax10.legend()



ax11 = fig.add_subplot(4, 3, 11)
tv = df_clean[df_clean["duration_unit"] == "Season"]["duration_value"]
ax11.hist(tv, bins=15, color="#8e44ad", edgecolor="white")
ax11.set_title("TV Show Seasons Distribution", fontsize=12, fontweight="bold")
ax11.set_xlabel("Number of Seasons")
ax11.set_ylabel("Count")



ax12 = fig.add_subplot(4, 3, 12)
sns.boxplot(y=df_clean["release_year"], color="#3498db", ax=ax12)
ax12.set_title("Outlier Detection — Release Year", fontsize=12, fontweight="bold")
ax12.set_ylabel("Release Year")


plt.tight_layout()
plt.savefig("Madhuri_Task1_DataCleaning.png", dpi=150, bbox_inches="tight")
plt.show()
print("Dashboard saved!")



print("\n" + "=" * 50)
print("         CLEANING SUMMARY REPORT")
print("=" * 50)
print(f"Original rows          : {df.shape[0]:,}")
print(f"Cleaned rows           : {df_clean.shape[0]:,}")
print(f"Duplicates removed     : {df.shape[0] - df_clean.shape[0]}")
print(f"Columns added          : year_added, month_added, duration_value, duration_unit")
print(f"Missing values before  : {df.isnull().sum().sum()}")
print(f"Missing values after   : {df_clean.isnull().sum().sum()}")
print(f"Outliers in release_yr : {len(outliers)}")
print(f"Total content          : {df_clean.shape[0]:,}")
print(f"Movies                 : {(df_clean['type']=='Movie').sum():,}")
print(f"TV Shows               : {(df_clean['type']=='Tv Show').sum():,}")
print(f"Top Country            : {df_clean['country'].value_counts().idxmax()}")
print(f"Most common rating     : {df_clean['rating'].value_counts().idxmax()}")