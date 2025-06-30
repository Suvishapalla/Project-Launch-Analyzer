# Project-Launch-Analyzer


A machine learning tool that predicts how successful a student tech project might be — based on its description, tags, and timing.

## Features
- NLP analysis of project idea descriptions
- TF-IDF vectorization + Logistic Regression
- Outputs success probability (0-100%)
- Configurable training on past project data

## Tech Stack
Python, Pandas, Scikit-learn, TF-IDF, Streamlit (optional)

## How to Run
1. Train model: `python model/train_model.py`
2. Run predictor: `python main.py`

## Files
- `main.py` → Input interface
- `train_model.py` → Builds and saves model
- `sample_projects.csv` → Training data

