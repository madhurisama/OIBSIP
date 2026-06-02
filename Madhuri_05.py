import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                              r2_score)
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("Housing.csv")

print("=" * 50)
print("HOUSE PRICE DATASET — OVERVIEW")
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


le = LabelEncoder()
cat_cols = df.select_dtypes(include="object").columns.tolist()
print(f"\nEncoding columns: {cat_cols}")
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

print(f"\n✔ Clean shape: {df.shape}")



print("\n" + "=" * 50)
print("DESCRIPTIVE STATISTICS")
print("=" * 50)
print(df.describe())
print(f"\nAverage House Price : ${df['price'].mean():,.2f}")
print(f"Median House Price  : ${df['price'].median():,.2f}")
print(f"Max House Price     : ${df['price'].max():,.2f}")
print(f"Min House Price     : ${df['price'].min():,.2f}")



X = df.drop("price", axis=1)
y = df["price"]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"\nTrain size : {X_train.shape[0]}")
print(f"Test size  : {X_test.shape[0]}")



print("\n" + "=" * 50)
print("MODEL TRAINING")
print("=" * 50)


lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)


lasso = Lasso(alpha=1.0)
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)


def evaluate(name, y_test, y_pred):
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    print(f"\n── {name} ──")
    print(f"  R² Score : {r2:.4f}")
    print(f"  RMSE     : ${rmse:,.2f}")
    print(f"  MAE      : ${mae:,.2f}")
    return r2, rmse, mae

lr_r2,    lr_rmse,    lr_mae    = evaluate("Linear Regression", y_test, lr_pred)
ridge_r2, ridge_rmse, ridge_mae = evaluate("Ridge Regression",  y_test, ridge_pred)
lasso_r2, lasso_rmse, lasso_mae = evaluate("Lasso Regression",  y_test, lasso_pred)

cv_scores = cross_val_score(lr, X_scaled, y, cv=5, scoring="r2")
print(f"\n5-Fold CV R² : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")


feature_importance = pd.Series(
    np.abs(lr.coef_), index=X.columns
).sort_values(ascending=False)

sns.set_theme(style="whitegrid")
fig = plt.figure(figsize=(22, 26))
fig.suptitle("House Price Prediction Dashboard",
             fontsize=22, fontweight="bold", y=1.01)



ax1 = fig.add_subplot(4, 3, 1)
ax1.hist(df["price"], bins=40,
         color="#3498db", edgecolor="white")
ax1.axvline(df["price"].mean(), color="red",
            linestyle="--",
            label=f"Mean: ${df['price'].mean():,.0f}")
ax1.set_title("House Price Distribution",
              fontsize=12, fontweight="bold")
ax1.set_xlabel("Price ($)")
ax1.set_ylabel("Count")
ax1.legend()



ax2 = fig.add_subplot(4, 3, 2)
corr = df.corr()
sns.heatmap(corr, annot=True, fmt=".2f",
            cmap="coolwarm", linewidths=0.5,
            ax=ax2, annot_kws={"size": 7})
ax2.set_title("Correlation Heatmap",
              fontsize=12, fontweight="bold")



ax3 = fig.add_subplot(4, 3, 3)
feature_importance.plot(kind="barh",
                        color="#9b59b6",
                        edgecolor="white", ax=ax3)
ax3.set_title("Feature Importance\n(Linear Regression)",
              fontsize=12, fontweight="bold")
ax3.set_xlabel("Absolute Coefficient")



ax4 = fig.add_subplot(4, 3, 4)
ax4.scatter(y_test, lr_pred, alpha=0.5,
            color="#2ecc71", edgecolors="white", s=30)
ax4.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color="red", linestyle="--", linewidth=2)
ax4.set_title(f"Actual vs Predicted\n"
              f"Linear Regression (R²={lr_r2:.3f})",
              fontsize=12, fontweight="bold")
ax4.set_xlabel("Actual Price")
ax4.set_ylabel("Predicted Price")



ax5 = fig.add_subplot(4, 3, 5)
ax5.scatter(y_test, ridge_pred, alpha=0.5,
            color="#e74c3c", edgecolors="white", s=30)
ax5.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color="blue", linestyle="--", linewidth=2)
ax5.set_title(f"Actual vs Predicted\n"
              f"Ridge Regression (R²={ridge_r2:.3f})",
              fontsize=12, fontweight="bold")
ax5.set_xlabel("Actual Price")
ax5.set_ylabel("Predicted Price")



ax6 = fig.add_subplot(4, 3, 6)
ax6.scatter(y_test, lasso_pred, alpha=0.5,
            color="#f39c12", edgecolors="white", s=30)
ax6.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color="blue", linestyle="--", linewidth=2)
ax6.set_title(f"Actual vs Predicted\n"
              f"Lasso Regression (R²={lasso_r2:.3f})",
              fontsize=12, fontweight="bold")
ax6.set_xlabel("Actual Price")
ax6.set_ylabel("Predicted Price")



ax7 = fig.add_subplot(4, 3, 7)
residuals = y_test - lr_pred
ax7.scatter(lr_pred, residuals, alpha=0.5,
            color="#3498db", edgecolors="white", s=30)
ax7.axhline(0, color="red", linestyle="--", linewidth=2)
ax7.set_title("Residual Plot — Linear Regression",
              fontsize=12, fontweight="bold")
ax7.set_xlabel("Predicted Price")
ax7.set_ylabel("Residuals")



ax8 = fig.add_subplot(4, 3, 8)
models  = ["Linear\nRegression", "Ridge\nRegression",
           "Lasso\nRegression"]
r2_scores = [lr_r2, ridge_r2, lasso_r2]
colors8 = ["#2ecc71", "#e74c3c", "#f39c12"]
bars8 = ax8.bar(models, r2_scores,
                color=colors8, edgecolor="white")
ax8.set_title("Model R² Score Comparison",
              fontsize=12, fontweight="bold")
ax8.set_ylabel("R² Score")
ax8.set_ylim(0, 1)
for bar, score in zip(bars8, r2_scores):
    ax8.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.01,
             f"{score:.4f}",
             ha="center", fontweight="bold")



ax9 = fig.add_subplot(4, 3, 9)
rmse_scores = [lr_rmse, ridge_rmse, lasso_rmse]
bars9 = ax9.bar(models, rmse_scores,
                color=colors8, edgecolor="white")
ax9.set_title("Model RMSE Comparison",
              fontsize=12, fontweight="bold")
ax9.set_ylabel("RMSE ($)")
for bar, score in zip(bars9, rmse_scores):
    ax9.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 100,
             f"${score:,.0f}",
             ha="center", fontsize=9,
             fontweight="bold")



ax10 = fig.add_subplot(4, 3, 10)
ax10.scatter(df["area"], df["price"],
             alpha=0.5, color="#1abc9c",
             edgecolors="white", s=30)
ax10.set_title("Price vs Area",
               fontsize=12, fontweight="bold")
ax10.set_xlabel("Area (sq ft)")
ax10.set_ylabel("Price ($)")



ax11 = fig.add_subplot(4, 3, 11)
sns.boxplot(x="bedrooms", y="price",
            data=df, palette="viridis", ax=ax11)
ax11.set_title("Price by Number of Bedrooms",
               fontsize=12, fontweight="bold")
ax11.set_xlabel("Bedrooms")
ax11.set_ylabel("Price ($)")

ax12 = fig.add_subplot(4, 3, 12)
ax12.plot(range(1, 6), cv_scores,
          marker="o", color="#e74c3c",
          linewidth=2)
ax12.axhline(cv_scores.mean(), color="blue",
             linestyle="--",
             label=f"Mean R²: {cv_scores.mean():.4f}")
ax12.set_title("5-Fold Cross Validation R² Scores",
               fontsize=12, fontweight="bold")
ax12.set_xlabel("Fold")
ax12.set_ylabel("R² Score")
ax12.set_xticks(range(1, 6))
ax12.legend()


plt.tight_layout()
plt.savefig("Madhuri_Task5_HousePricePrediction.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("✅ Dashboard saved!")



print("\n" + "=" * 50)
print("         KEY FINDINGS")
print("=" * 50)
print(f"Total Houses         : {len(df):,}")
print(f"Average Price        : ${df['price'].mean():,.2f}")
print(f"Median Price         : ${df['price'].median():,.2f}")
print(f"\nModel Performance:")
print(f"  Linear Regression  : R²={lr_r2:.4f}  RMSE=${lr_rmse:,.2f}")
print(f"  Ridge Regression   : R²={ridge_r2:.4f}  RMSE=${ridge_rmse:,.2f}")
print(f"  Lasso Regression   : R²={lasso_r2:.4f}  RMSE=${lasso_rmse:,.2f}")
print(f"\n5-Fold CV R²         : {cv_scores.mean():.4f}")
print(f"Top Feature          : {feature_importance.idxmax()}")
print(f"\n── Insights ──")
print("1. Area is the strongest predictor of price")
print("2. More bedrooms generally means higher price")
print("3. All 3 models perform similarly on this dataset")
print("4. Ridge and Lasso help reduce overfitting")