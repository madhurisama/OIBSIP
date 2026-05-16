import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")



df = pd.read_csv("retail_sales_dataset.csv")

print("=" * 50)
print("RETAIL SALES DATASET — OVERVIEW")
print("=" * 50)
print(f"Shape      : {df.shape}")
print(f"Columns    : {list(df.columns)}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nData Types:\n{df.dtypes}")



print("\n" + "=" * 50)
print("DATA CLEANING")
print("=" * 50)

print(f"Missing values:\n{df.isnull().sum()}")
print(f"Duplicates: {df.duplicated().sum()}")

df.drop_duplicates(inplace=True)
df.dropna(inplace=True)


df["Date"] = pd.to_datetime(df["Date"])

df["Year"]    = df["Date"].dt.year
df["Month"]   = df["Date"].dt.month
df["Month_Name"] = df["Date"].dt.strftime("%b")
df["Day"]     = df["Date"].dt.day
df["Weekday"] = df["Date"].dt.day_name()

print(f"\n✔ Date converted and time features extracted")
print(f"✔ Clean shape: {df.shape}")



print("\n" + "=" * 50)
print("DESCRIPTIVE STATISTICS")
print("=" * 50)
print(df.describe())

print(f"\nTotal Revenue       : ${df['Total Amount'].sum():,.2f}")
print(f"Average Transaction : ${df['Total Amount'].mean():,.2f}")
print(f"Max Transaction     : ${df['Total Amount'].max():,.2f}")
print(f"Min Transaction     : ${df['Total Amount'].min():,.2f}")
print(f"Median Transaction  : ${df['Total Amount'].median():,.2f}")
print(f"Std Deviation       : ${df['Total Amount'].std():,.2f}")



monthly_sales = df.groupby(["Year", "Month", "Month_Name"])["Total Amount"].sum().reset_index()
monthly_sales = monthly_sales.sort_values(["Year", "Month"])


category_sales = df.groupby("Product Category")["Total Amount"].sum().sort_values(ascending=False)


gender_sales = df.groupby("Gender")["Total Amount"].sum()


df["Age Group"] = pd.cut(df["Age"],
                          bins=[0, 25, 35, 45, 55, 100],
                          labels=["18-25", "26-35", "36-45", "46-55", "55+"])
age_sales = df.groupby("Age Group")["Total Amount"].sum()


weekday_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
weekday_sales = df.groupby("Weekday")["Total Amount"].sum().reindex(weekday_order)


product_qty = df.groupby("Product Category")["Quantity"].sum().sort_values(ascending=False)


avg_price = df.groupby("Product Category")["Price per Unit"].mean().sort_values(ascending=False)



sns.set_theme(style="whitegrid")
fig = plt.figure(figsize=(22, 26))
fig.suptitle("Retail Sales EDA Dashboard ",
             fontsize=22, fontweight="bold", y=1.01)



ax1 = fig.add_subplot(4, 3, 1)
ax1.plot(range(len(monthly_sales)),
         monthly_sales["Total Amount"],
         marker="o", color="#3498db", linewidth=2)
ax1.fill_between(range(len(monthly_sales)),
                 monthly_sales["Total Amount"],
                 alpha=0.3, color="#3498db")
ax1.set_title("Monthly Sales Trend", fontsize=12, fontweight="bold")
ax1.set_ylabel("Total Revenue ($)")
ax1.set_xlabel("Month")
ax1.set_xticks(range(len(monthly_sales)))
ax1.set_xticklabels(monthly_sales["Month_Name"], rotation=45, ha="right")



ax2 = fig.add_subplot(4, 3, 2)
colors2 = sns.color_palette("viridis", len(category_sales))
bars = ax2.bar(category_sales.index, category_sales.values,
               color=colors2, edgecolor="white")
ax2.set_title("Revenue by Product Category", fontsize=12, fontweight="bold")
ax2.set_ylabel("Total Revenue ($)")
ax2.set_xticklabels(category_sales.index, rotation=30, ha="right")
for bar in bars:
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 100,
             f"${bar.get_height():,.0f}",
             ha="center", fontsize=8, fontweight="bold")


ax3 = fig.add_subplot(4, 3, 3)
ax3.pie(gender_sales.values, labels=gender_sales.index,
        autopct="%1.1f%%",
        colors=["#3498db", "#e91e8c"],
        startangle=90)
ax3.set_title("Revenue by Gender", fontsize=12, fontweight="bold")



ax4 = fig.add_subplot(4, 3, 4)
ax4.bar(age_sales.index.astype(str), age_sales.values,
        color="#e67e22", edgecolor="white")
ax4.set_title("Revenue by Age Group", fontsize=12, fontweight="bold")
ax4.set_xlabel("Age Group")
ax4.set_ylabel("Total Revenue ($)")
for i, v in enumerate(age_sales.values):
    ax4.text(i, v + 100, f"${v:,.0f}",
             ha="center", fontsize=8, fontweight="bold")



ax5 = fig.add_subplot(4, 3, 5)
colors5 = ["#2ecc71" if v == weekday_sales.max()
           else "#e74c3c" if v == weekday_sales.min()
           else "#95a5a6" for v in weekday_sales.values]
ax5.bar(weekday_sales.index, weekday_sales.values,
        color=colors5, edgecolor="white")
ax5.set_title("Revenue by Weekday", fontsize=12, fontweight="bold")
ax5.set_ylabel("Total Revenue ($)")
ax5.set_xticklabels(weekday_sales.index, rotation=45, ha="right")



ax6 = fig.add_subplot(4, 3, 6)
ax6.barh(product_qty.index, product_qty.values,
         color="#9b59b6", edgecolor="white")
ax6.set_title("Quantity Sold by Category", fontsize=12, fontweight="bold")
ax6.set_xlabel("Total Quantity")
for i, v in enumerate(product_qty.values):
    ax6.text(v + 1, i, str(v), va="center", fontsize=9)



ax7 = fig.add_subplot(4, 3, 7)
ax7.hist(df["Price per Unit"], bins=30,
         color="#1abc9c", edgecolor="white")
ax7.axvline(df["Price per Unit"].mean(), color="red",
            linestyle="--", label=f"Mean: ${df['Price per Unit'].mean():.2f}")
ax7.set_title("Price per Unit Distribution", fontsize=12, fontweight="bold")
ax7.set_xlabel("Price ($)")
ax7.set_ylabel("Count")
ax7.legend()



ax8 = fig.add_subplot(4, 3, 8)
ax8.hist(df["Total Amount"], bins=40,
         color="#e74c3c", edgecolor="white", alpha=0.8)
ax8.axvline(df["Total Amount"].mean(), color="blue",
            linestyle="--", label=f"Mean: ${df['Total Amount'].mean():.2f}")
ax8.set_title("Transaction Amount Distribution", fontsize=12, fontweight="bold")
ax8.set_xlabel("Amount ($)")
ax8.set_ylabel("Count")
ax8.legend()


ax9 = fig.add_subplot(4, 3, 9)
ax9.bar(avg_price.index, avg_price.values,
        color="#f39c12", edgecolor="white")
ax9.set_title("Avg Price per Unit by Category", fontsize=12, fontweight="bold")
ax9.set_ylabel("Avg Price ($)")
ax9.set_xticklabels(avg_price.index, rotation=30, ha="right")
for i, v in enumerate(avg_price.values):
    ax9.text(i, v + 0.5, f"${v:.2f}",
             ha="center", fontsize=9, fontweight="bold")



ax10 = fig.add_subplot(4, 3, 10)
num_cols = ["Age", "Quantity", "Price per Unit", "Total Amount"]
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, ax=ax10)
ax10.set_title("Correlation Heatmap", fontsize=12, fontweight="bold")



ax11 = fig.add_subplot(4, 3, 11)
ax11.hist(df["Age"], bins=20, color="#8e44ad",
          edgecolor="white")
ax11.axvline(df["Age"].mean(), color="red", linestyle="--",
             label=f"Mean Age: {df['Age'].mean():.1f}")
ax11.set_title("Customer Age Distribution", fontsize=12, fontweight="bold")
ax11.set_xlabel("Age")
ax11.set_ylabel("Count")
ax11.legend()



ax12 = fig.add_subplot(4, 3, 12)
pivot = df.pivot_table(values="Total Amount",
                       index="Gender",
                       columns="Product Category",
                       aggfunc="sum")
sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd",
            linewidths=0.5, ax=ax12)
ax12.set_title("Revenue: Gender vs Category", fontsize=12, fontweight="bold")


plt.tight_layout()
plt.savefig("Madhuri_Task2_RetailSalesEDA.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Dashboard saved!")



print("\n" + "=" * 50)
print("      KEY FINDINGS & RECOMMENDATIONS")
print("=" * 50)
print(f"Total Revenue           : ${df['Total Amount'].sum():,.2f}")
print(f"Total Transactions      : {len(df):,}")
print(f"Best Category           : {category_sales.idxmax()} (${category_sales.max():,.2f})")
print(f"Worst Category          : {category_sales.idxmin()} (${category_sales.min():,.2f})")
print(f"Best Weekday            : {weekday_sales.idxmax()}")
print(f"Most Active Age Group   : {age_sales.idxmax()}")
print(f"Top Gender by Revenue   : {gender_sales.idxmax()}")
print(f"Avg Transaction Value   : ${df['Total Amount'].mean():,.2f}")

print("\n── Recommendations ──")
print("1. Focus marketing on best-performing category")
print("2. Run weekend promotions on slow weekdays")
print("3. Target 26-45 age group with personalized offers")
print("4. Introduce loyalty programs for top customers")
print("5. Optimize inventory for high-demand products")