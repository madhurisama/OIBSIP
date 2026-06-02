import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re
import nltk
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (classification_report,
                             confusion_matrix, accuracy_score)
import warnings
warnings.filterwarnings("ignore")


cols = ["TweetID", "Entity", "Sentiment", "Text"]

try:
    train = pd.read_csv("twitter_training.csv", header=None, names=cols)
    val   = pd.read_csv("twitter_validation.csv", header=None, names=cols)
    df    = pd.concat([train, val], ignore_index=True)
    print("Loaded training + validation")
except FileNotFoundError:
    df = pd.read_csv("twitter_training.csv", header=None, names=cols)
    print("Loaded training only")

print(f"Shape     : {df.shape}")
print(f"Sentiments: {df['Sentiment'].unique()}")
print(f"\nFirst 5 rows:\n{df.head()}")



print("\n" + "=" * 50)
print("DATA CLEANING")
print("=" * 50)

df.dropna(subset=["Text", "Sentiment"], inplace=True)
df = df[df["Sentiment"].isin(["Positive","Negative","Neutral","Irrelevant"])]

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["Clean_Text"] = df["Text"].apply(clean_text)
df["Word_Count"] = df["Clean_Text"].apply(lambda x: len(x.split()))

print(f"Clean shape: {df.shape}")
print(f"\nSentiment counts:\n{df['Sentiment'].value_counts()}")



STOPWORDS = set(stopwords.words("english"))
STOPWORDS.update(["im","dont","cant","just","like",
                   "get","one","would","also","us","amp"])

def get_words(sentiment):
    texts = " ".join(df[df["Sentiment"] == sentiment]["Clean_Text"])
    words = [w for w in texts.split()
             if w not in STOPWORDS and len(w) > 2]
    return words

pos_words = get_words("Positive")
neg_words = get_words("Negative")
neu_words = get_words("Neutral")

top_entities = df["Entity"].value_counts().head(10).index.tolist()
df_top = df[df["Entity"].isin(top_entities)]


print("\n" + "=" * 50)
print("MACHINE LEARNING MODELS")
print("=" * 50)


df_ml = df[df["Sentiment"].isin(["Positive","Negative","Neutral"])].copy()

X = df_ml["Clean_Text"]
y = df_ml["Sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)


nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
nb_pred  = nb_model.predict(X_test_tfidf)
nb_acc   = accuracy_score(y_test, nb_pred)


svc_model = LinearSVC(random_state=42, max_iter=1000)
svc_model.fit(X_train_tfidf, y_train)
svc_pred  = svc_model.predict(X_test_tfidf)
svc_acc   = accuracy_score(y_test, svc_pred)

print(f"Naive Bayes Accuracy : {nb_acc*100:.2f}%")
print(f"Linear SVC Accuracy  : {svc_acc*100:.2f}%")


best_pred  = svc_pred if svc_acc > nb_acc else nb_pred
best_model = "Linear SVC" if svc_acc > nb_acc else "Naive Bayes"
best_acc   = max(svc_acc, nb_acc)
print(f"\nBest Model: {best_model} ({best_acc*100:.2f}%)")
print(f"\nClassification Report ({best_model}):")
print(classification_report(y_test, best_pred))



sns.set_theme(style="whitegrid")
COLORS = {
    "Positive"   : "#2ecc71",
    "Negative"   : "#e74c3c",
    "Neutral"    : "#3498db",
    "Irrelevant" : "#95a5a6"
}

fig = plt.figure(figsize=(22, 26))
fig.suptitle("Sentiment Analysis Dashboard — Madhuri",
             fontsize=22, fontweight="bold", y=1.01)



ax1 = fig.add_subplot(4, 3, 1)
sent_counts  = df["Sentiment"].value_counts()
wedge_colors = [COLORS[s] for s in sent_counts.index]
ax1.pie(sent_counts.values, labels=sent_counts.index,
        colors=wedge_colors, autopct="%1.1f%%", startangle=140)
ax1.set_title("Overall Sentiment Distribution",
              fontsize=12, fontweight="bold")



ax2 = fig.add_subplot(4, 3, 2)
bars = ax2.bar(sent_counts.index, sent_counts.values,
               color=wedge_colors, edgecolor="white")
ax2.set_title("Sentiment Count", fontsize=12, fontweight="bold")
ax2.set_ylabel("Number of Tweets")
for bar in bars:
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 20,
             f"{int(bar.get_height()):,}",
             ha="center", fontsize=9, fontweight="bold")



ax3 = fig.add_subplot(4, 3, 3)
pivot = (df_top.groupby(["Entity","Sentiment"])
               .size().unstack(fill_value=0))
pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
pivot_pct = pivot_pct[[c for c in
             ["Positive","Negative","Neutral","Irrelevant"]
             if c in pivot_pct.columns]]
pivot_pct.plot(kind="barh", stacked=True,
               color=[COLORS[c] for c in pivot_pct.columns],
               edgecolor="white", linewidth=0.5, ax=ax3)
ax3.set_title("Sentiment % by Brand", fontsize=12, fontweight="bold")
ax3.set_xlabel("Percentage (%)")
ax3.legend(loc="lower right", fontsize=8)



ax4 = fig.add_subplot(4, 3, 4)
wc_pos = WordCloud(width=600, height=300,
                   background_color="white",
                   colormap="Greens", max_words=80,
                   stopwords=STOPWORDS
                   ).generate(" ".join(pos_words))
ax4.imshow(wc_pos, interpolation="bilinear")
ax4.axis("off")
ax4.set_title("Word Cloud — Positive",
              fontsize=12, fontweight="bold", color="#2ecc71")



ax5 = fig.add_subplot(4, 3, 5)
wc_neg = WordCloud(width=600, height=300,
                   background_color="white",
                   colormap="Reds", max_words=80,
                   stopwords=STOPWORDS
                   ).generate(" ".join(neg_words))
ax5.imshow(wc_neg, interpolation="bilinear")
ax5.axis("off")
ax5.set_title("Word Cloud — Negative",
              fontsize=12, fontweight="bold", color="#e74c3c")



ax6 = fig.add_subplot(4, 3, 6)
wc_neu = WordCloud(width=600, height=300,
                   background_color="white",
                   colormap="Blues", max_words=80,
                   stopwords=STOPWORDS
                   ).generate(" ".join(neu_words))
ax6.imshow(wc_neu, interpolation="bilinear")
ax6.axis("off")
ax6.set_title("Word Cloud — Neutral",
              fontsize=12, fontweight="bold", color="#3498db")



ax7 = fig.add_subplot(4, 3, 7)
top_pos = Counter(pos_words).most_common(15)
words_p, counts_p = zip(*top_pos)
ax7.barh(words_p[::-1], counts_p[::-1],
         color="#2ecc71", edgecolor="white")
ax7.set_title("Top 15 Positive Words",
              fontsize=12, fontweight="bold")
ax7.set_xlabel("Frequency")


ax8 = fig.add_subplot(4, 3, 8)
top_neg = Counter(neg_words).most_common(15)
words_n, counts_n = zip(*top_neg)
ax8.barh(words_n[::-1], counts_n[::-1],
         color="#e74c3c", edgecolor="white")
ax8.set_title("Top 15 Negative Words",
              fontsize=12, fontweight="bold")
ax8.set_xlabel("Frequency")



ax9 = fig.add_subplot(4, 3, 9)
sns.boxplot(x="Sentiment", y="Word_Count",
            data=df[df["Word_Count"] < 60],
            palette=COLORS,
            order=["Positive","Negative",
                   "Neutral","Irrelevant"],
            ax=ax9)
ax9.set_title("Tweet Length by Sentiment",
              fontsize=12, fontweight="bold")
ax9.set_ylabel("Word Count")
ax9.set_xlabel("")



ax10 = fig.add_subplot(4, 3, 10)
top10 = df["Entity"].value_counts().head(10)
ax10.barh(top10.index[::-1], top10.values[::-1],
          color="#9b59b6", edgecolor="white")
ax10.set_title("Top 10 Brands by Tweet Volume",
               fontsize=12, fontweight="bold")
ax10.set_xlabel("Tweet Count")



ax11 = fig.add_subplot(4, 3, 11)
models  = ["Naive Bayes", "Linear SVC"]
scores  = [nb_acc * 100, svc_acc * 100]
colors11 = ["#f39c12", "#3498db"]
bars11  = ax11.bar(models, scores,
                   color=colors11, edgecolor="white")
ax11.set_title("Model Accuracy Comparison",
               fontsize=12, fontweight="bold")
ax11.set_ylabel("Accuracy (%)")
ax11.set_ylim(0, 100)
for bar, score in zip(bars11, scores):
    ax11.text(bar.get_x() + bar.get_width()/2,
              bar.get_height() + 0.5,
              f"{score:.2f}%",
              ha="center", fontweight="bold")



ax12 = fig.add_subplot(4, 3, 12)
cm = confusion_matrix(y_test, best_pred,
                      labels=["Positive","Negative","Neutral"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Positive","Negative","Neutral"],
            yticklabels=["Positive","Negative","Neutral"],
            ax=ax12)
ax12.set_title(f"Confusion Matrix — {best_model}",
               fontsize=12, fontweight="bold")
ax12.set_ylabel("Actual")
ax12.set_xlabel("Predicted")


plt.tight_layout()
plt.savefig("Madhuri_Task4_SentimentAnalysis.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("✅ Dashboard saved!")



print("\n" + "=" * 50)
print("         KEY FINDINGS")
print("=" * 50)
print(f"Total Tweets Analyzed : {len(df):,}")
print(f"Unique Brands         : {df['Entity'].nunique()}")
print(f"\nSentiment Breakdown:")
for s, c in df["Sentiment"].value_counts().items():
    print(f"  {s:<12}: {c:>6,} ({c/len(df)*100:.1f}%)")
print(f"\nBest ML Model  : {best_model}")
print(f"Best Accuracy  : {best_acc*100:.2f}%")
print(f"Naive Bayes    : {nb_acc*100:.2f}%")
print(f"Linear SVC     : {svc_acc*100:.2f}%")
print(f"\nTop 5 Positive Words: "
      f"{[w for w,_ in Counter(pos_words).most_common(5)]}")
print(f"Top 5 Negative Words: "
      f"{[w for w,_ in Counter(neg_words).most_common(5)]}")