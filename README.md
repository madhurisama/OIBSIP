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