# OIBSIP — Data Analytics Internship
## Task 1: Data Cleaning — Netflix Movies & TV Shows Dataset

### Objective
Clean and analyze the Netflix dataset by handling missing values,
removing duplicates, standardizing formats, and detecting outliers.

### Dataset
Netflix Movies and TV Shows — Kaggle

### Steps Performed
- Removed duplicate records
- Filled missing values (director, cast, country, rating, duration)
- Standardized text columns (strip, title case, upper case)
- Converted date_added to datetime format
- Extracted year_added and month_added features
- Split duration into duration_value and duration_unit
- Detected outliers in release_year using IQR method

### Key Findings
- No missing values after cleaning
- United States is the top content-producing country
- Movies dominate over TV Shows (~70% vs ~30%)
- Content additions peaked around 2019-2020
- Average movie duration is ~100 minutes

### Libraries Used
pandas, numpy, matplotlib, seaborn, missingno

### Output
- netflix_cleaned.csv — cleaned dataset
- Madhuri_Task1_DataCleaning.png — analysis dashboard

---

## Task 2: EDA on Retail Sales Data

### Objective
Perform Exploratory Data Analysis (EDA) on a retail sales dataset to uncover
patterns, trends, and insights that help the retail business make informed decisions.

### Dataset
Retail Sales Dataset — Kaggle
(mohammadtalib786/retail-sales-dataset)

### Steps Performed
- Loaded and explored the dataset structure
- Removed duplicates and handled missing values
- Converted Date column to datetime format
- Extracted time features (Year, Month, Weekday)
- Created Age Group categories using binning
- Calculated descriptive statistics (mean, median, mode, std)
- Performed time series analysis on monthly sales trends
- Analyzed customer demographics and purchasing behavior
- Built a 12-plot EDA dashboard

### Key Findings
- Identified best performing product category by revenue
- Found most active customer age group (26-45)
- Discovered best and worst performing weekdays for sales
- Analyzed gender-based purchasing patterns
- Uncovered seasonal trends in monthly sales

### Recommendations
1. Focus marketing on best-performing product category
2. Run promotions on slow weekdays to boost sales
3. Target 26-45 age group with personalized offers
4. Introduce loyalty programs for top customers
5. Optimize inventory for high-demand products

### Libraries Used
pandas, numpy, matplotlib, seaborn

### Output
- Madhuri_Task2_RetailSalesEDA.png — EDA dashboard

- ---

## Task 3: Customer Segmentation Analysis

### Objective
Group customers into distinct segments using K-Means clustering
based on Annual Income and Spending Score.

### Dataset
Mall Customers Dataset — Kaggle
(vjchoudhary7/customer-segmentation-tutorial-in-python)

### Steps Performed
- Data cleaning and preprocessing
- Gender encoding
- Feature scaling using StandardScaler
- Optimal K selection using Elbow Method + Silhouette Score
- K-Means clustering with K=5
- Cluster profiling and labeling
- 12-plot visualization dashboard

### Key Findings
- 5 distinct customer segments identified
- VIP customers: High Income + High Spending
- Budget customers: Low Income + Low Spending
- Potential targets: High Income + Low Spending
- Loyal risk group: Low Income + High Spending

### Libraries Used
pandas, numpy, matplotlib, seaborn, scikit-learn

### Output
- Madhuri_Task3_CustomerSegmentation.png — segmentation dashboard
 ## Task 4: Sentiment Analysis

### Objective
Develop a sentiment analysis model to
accurately classify text data as Positive,
Negative or Neutral, providing valuable
insights into public opinion and customer
feedback.

### Dataset
Twitter Entity Sentiment Analysis — Kaggle
- twitter_training.csv
- twitter_validation.csv

### Steps Performed
- Loaded and merged training + validation data
- Cleaned tweet text:
  - Removed URLs, mentions, hashtags
  - Removed special characters and numbers
  - Converted to lowercase
- Extracted word count feature
- Built word frequency analysis
- Applied TF-IDF Vectorization (bigrams)
- Trained and compared 2 ML models
- Generated Word Clouds for each sentiment
- Built 12-plot visualization dashboard

### Models Used
- Multinomial Naive Bayes
- Linear Support Vector Classifier (SVC)

### Key Findings
- Linear SVC outperforms Naive Bayes
- Most tweets showed Positive sentiment
- Identified top positive and negative words
- Word clouds revealed key sentiment drivers
- Analyzed sentiment patterns across brands

### Visualizations
- Overall sentiment distribution
- Sentiment % by brand (Stacked bar)
- Word Clouds (Positive, Negative, Neutral)
- Top 15 Positive and Negative words
- Tweet length by sentiment (Boxplot)
- Top 10 brands by tweet volume
- Model accuracy comparison
- Confusion matrix

### Libraries Used
pandas, numpy, matplotlib, seaborn,
scikit-learn, nltk, wordcloud

### Output
- Madhuri_Task4_SentimentAnalysis.png
## Task 5: House Price Prediction

### Objective
Build a predictive model using Linear
Regression to estimate house prices based
on relevant features like area, bedrooms,
and other property characteristics.

### Dataset
Housing Prices Dataset — Kaggle
(yasserh/housing-prices-dataset)

### Steps Performed
- Data cleaning and preprocessing
- Label encoding of categorical columns
- Feature scaling using StandardScaler
- Train/test split (80/20)
- Trained 3 regression models
- Evaluated using R², RMSE, MAE
- 5-Fold Cross Validation
- Feature importance analysis
- Built 12-plot visualization dashboard

### Models Used
- Linear Regression
- Ridge Regression (alpha=1.0)
- Lasso Regression (alpha=1.0)

### Key Findings
- Area is the strongest price predictor
- More bedrooms generally means higher price
- All 3 models perform similarly
- Ridge and Lasso help reduce overfitting
- 5-Fold CV confirms model consistency

### Visualizations
- Price distribution histogram
- Correlation heatmap
- Feature importance bar chart
- Actual vs Predicted (all 3 models)
- Residual plot
- Model R² comparison
- RMSE comparison
- Price vs Area scatter plot
- Price by bedrooms boxplot
- Cross validation scores

### Libraries Used
pandas, numpy, matplotlib, seaborn,
scikit-learn

### Output
- Madhuri_Task5_HousePricePrediction.png
