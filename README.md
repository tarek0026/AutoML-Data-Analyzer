🚀 AutoML Data Analyzer

A powerful Streamlit-based AutoML dashboard for fast data exploration, clustering, model comparison, and automated insight generation.

📌 Overview

AutoML Data Analyzer is a lightweight yet production-oriented analytics tool designed to help you understand any tabular dataset .

Upload your CSV file and instantly:

📊 Explore your data
🧹 Clean & preprocess it automatically
🔍 Discover hidden patterns with clustering
🤖 Train ML models (if target exists)
💡 Generate real, actionable insights

Built with clean architecture to showcase real-world ML pipeline design — not just experiments.

✨ Features
📂 Data Handling
Upload CSV datasets مباشرة من الـ UI
Automatic data validation & cleaning
Smart handling for:
Missing values
Encoding categorical features
Feature scaling

🔍 Clustering Analysis
⚡ Supports:
K-Means
DBSCAN
📈 Automatic evaluation & comparison
🧠 Intelligent selection of best clustering approach
🤖 Machine Learning (Optional)

If a target column is provided:

Models included:
Logistic Regression
Random Forest
Gradient Boosting
🔥 Auto model selection based on performance
📊 Feature importance extraction
🎯 Clean evaluation metrics
💡 Insight Generation
🧠 Automated business-style insights
📌 Key feature impact explanations
📉 Honest performance interpretation:
Weak
Moderate
Strong
📊 Visualization
Interactive dashboards using Plotly
Clean and minimal UI via Streamlit
Data distributions & patterns visualization
🛠️ Tech Stack
🐍 Python
⚡ Streamlit
📊 pandas
🔢 NumPy
🤖 scikit-learn
📈 Plotly
▶️ How to Run
1️⃣ Setup Environment
pip install -r requirements.txt
2️⃣ Run the App
streamlit run app/app.py
3️⃣ Start Exploring
Upload your dataset 📂
Choose:
🔍 Analysis
🤖 Modeling
Get insights instantly 💡
📁 Project Structure
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
⚠️ Notes
PCA is used only for clustering visualization (not model training)
Project focuses on real pipeline design, not just notebooks
Designed as an analysis tool, not a deployment API
🎯 Why This Project?

This project demonstrates:

End-to-end ML pipeline design
Clean modular architecture
Real-world AutoML thinking
Practical Streamlit app deployment
👨‍💻 Author

Built as a production-ready portfolio project for:

Applied Machine Learning
Data Analysis Automation
Streamlit App Development
