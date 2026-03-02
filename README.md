# 📊 Superstore Sales Analysis & Forecasting

## 📌 Project Overview

This project performs end-to-end sales analysis and time series forecasting using a Superstore-style retail dataset.

It includes:
- Data validation & preprocessing
- Exploratory Data Analysis (EDA)
- Monthly sales trend analysis
- ARIMA-based sales forecasting
- Automated testing & validation checks

The system automatically generates a sample dataset if none is provided.

---

## 🎯 Objectives

- Analyze category-wise sales performance
- Identify most profitable regions
- Visualize monthly sales trends
- Forecast future sales for inventory planning
- Ensure reliability using automated test assertions

---

## 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Statsmodels (ARIMA)
- Time Series Forecasting

---

## 📊 Exploratory Data Analysis

### 1️⃣ Category-wise Sales
- Identifies highest revenue-generating category

### 2️⃣ Region-wise Profit
- Determines most profitable region

### 3️⃣ Monthly Sales Trend
- Resampled monthly sales
- Time-series visualization

---

## 🔮 Forecasting Model

### Model Used:
ARIMA (Auto Regressive Integrated Moving Average)

- Multiple ARIMA configurations tested
- Best model selected based on lowest AIC
- 6-month future sales forecast generated
- Naive fallback model if ARIMA fails

Forecast visualization:
- Historical sales
- Predicted future sales

---

## 🧠 Business Insights

- Identifies best performing product categories
- Determines most profitable region
- Enables data-driven inventory planning
- Supports marketing strategy decisions

---

## ✅ Automated Validations

Includes built-in tests to ensure:

- Required columns exist
- Monthly data is valid
- Forecast has correct length
- No missing values in predictions
- Forecast dates follow historical timeline

---

## 📂 How to Run

```bash
python main.py
