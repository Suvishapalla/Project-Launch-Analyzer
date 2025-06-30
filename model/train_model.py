import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
from utils.preprocess import preprocess_text

# Load and clean data
df = pd.read_csv("data/sample_projects.csv")
df["clean_text"] = df["description"].apply(preprocess_text)

# TF-IDF vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["clean_text"])
y = df["label"]

# Train logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Save model and vectorizer
joblib.dump(model, "model/model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

