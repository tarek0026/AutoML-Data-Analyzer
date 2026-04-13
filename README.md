# AutoML Data Analyzer

A Streamlit-based AutoML dashboard for exploratory data analysis, clustering, model comparison, and insight generation.

## Overview

AutoML Data Analyzer is designed as a lightweight analytics application for quickly understanding tabular datasets. Users can upload a CSV file, review dataset quality, explore feature distributions, run clustering, and optionally train a supervised machine learning model when a target column is available.

The project is structured for readability and portfolio presentation, with the app entry point separated from reusable pipeline code in `src/`.

## Features

- Upload and inspect CSV datasets in a Streamlit dashboard
- Automatic preprocessing with missing-value handling, encoding, and scaling
- Clustering workflow with KMeans and DBSCAN comparison
- Optional supervised modeling with:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
- Best-model selection based on evaluation score
- Feature importance visualization for top drivers
- Business-style insights and recommendations
- Honest performance interpretation using weak, moderate, and strong quality bands

## Tech Stack

- Python
- Streamlit
- pandas
- NumPy
- scikit-learn
- Plotly

## How to Run

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the app:

```bash
streamlit run app/app.py
```

4. Upload a CSV file in the sidebar and run analysis or modeling.

## Project Structure

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

## Notes

- PCA is used only for clustering visualization, not for model training.
- The repository is intentionally kept focused on the production app path.
- The app is designed as an analysis and insights tool rather than a prediction-serving product.

## Author

Prepared as a production-ready portfolio project for applied machine learning and Streamlit dashboard development.
