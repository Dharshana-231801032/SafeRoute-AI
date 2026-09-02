# SafeRoute AI 🚦

**Intelligent Road Accident Severity Prediction and Geospatial Hotspot Risk Decision Support System**

## Overview
SafeRoute AI is an end-to-end machine learning application that:
- Predicts road accident severity (1–4) using Random Forest, XGBoost, and LightGBM
- Detects accident hotspots using DBSCAN clustering
- Visualizes insights through an interactive Streamlit dashboard

## Datasets
- **US Accidents** (7.7M records, 2016–2023) — Primary ML dataset
- **Indian MoRTH Reports** (2008–2024) — Comparative analysis

## Tech Stack
Python 3.11 | Scikit-learn | XGBoost | LightGBM | SHAP | DBSCAN | Streamlit | GeoPandas | Folium | Polars

## Project Structure
SafeRoute-AI/
├── notebooks/   ← EDA & experiments (Jupyter)
├── src/         ← Python modules
├── app/         ← Streamlit dashboard
├── models/      ← Trained model files (local only)
├── data/        ← Raw datasets (local only)
└── reports/     ← Figures & charts

## Setup
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt