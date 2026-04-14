# 🚀 AutoML Data Analyzer

🔗 Live Demo:
👉 https://automl-data-analyzer-gg3atmy2dakotgrwzsbixk.streamlit.app/

A powerful **Streamlit-based AutoML dashboard** for fast data exploration, clustering, model comparison, and automated insight generation.

---

## 📌 Overview

**AutoML Data Analyzer** is a lightweight yet production-oriented analytics tool designed to help you understand any tabular dataset quickly and efficiently.

Upload your CSV file and instantly:

* 📊 Explore your data
* 🧹 Clean & preprocess automatically
* 🔍 Discover patterns with clustering
* 🤖 Train ML models (if target exists)
* 💡 Generate actionable insights

Built with clean architecture to showcase real-world ML pipeline design.

---

## ✨ Features

### 📂 Data Handling

* Upload CSV datasets directly from UI
* Automatic data validation & cleaning
* Handles:

  * Missing values
  * Encoding categorical features
  * Feature scaling

---

### 🔍 Clustering Analysis

* Supports:

  * K-Means
  * DBSCAN
* Automatic evaluation & comparison
* Intelligent best-model selection

---

### 🤖 Machine Learning (Optional)

If a target column is provided:

* Models:

  * Logistic Regression
  * Random Forest
  * Gradient Boosting

* Auto model selection

* Feature importance extraction

* Clean evaluation metrics

---

### 💡 Insight Generation

* Automated business-style insights
* Feature impact explanations
* Honest performance interpretation:

  * Weak
  * Moderate
  * Strong

---

### 📊 Visualization

* Interactive dashboards (Plotly)
* Clean UI with Streamlit
* Data distributions & patterns

---

## 🛠️ Tech Stack

* Python
* Streamlit
* pandas
* NumPy
* scikit-learn
* Plotly

---

## ▶️ How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run app/app.py
```

### 3. Start

* Upload your dataset
* Choose analysis or modeling
* Get insights instantly

---

## 📁 Project Structure

```text
AutoML-Data-Analyzer/
├── .streamlit/
│   └── config.toml
├── app/
│   └── app.py
├── data/
│   └── raw/
│       └── sample_data.csv
├── src/
│   ├── clustering/
│   │   └── clustering_pipeline.py
│   ├── data_processing/
│   │   ├── data_validation.py
│   │   └── preprocessing_pipeline.py
│   ├── insights/
│   │   ├── business_insights.py
│   │   └── insights_pipeline.py
│   ├── modeling/
│   │   └── modeling_pipeline.py
│   ├── visualization/
│   │   └── dashboard_viz.py
│   └── pipeline_orchestrator.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## ⚠️ Notes

* PCA is used only for clustering visualization
* Focused on production-style pipeline
* Designed as an analysis tool, not a deployment API

---

## 🎯 Why This Project?

* End-to-end ML pipeline
* Clean modular architecture
* Real AutoML workflow
* Practical Streamlit deployment

---

## 👨‍💻 Author

Production-ready portfolio project for:

* Applied Machine Learning
* Data Analysis Automation
* Streamlit Development
