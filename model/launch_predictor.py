import joblib

def load_model_and_vectorizer():
    model = joblib.load("model/model.pkl")
    vectorizer = joblib.load("model/vectorizer.pkl")
    return model, vectorizer

def predict(model, vectorizer, input_text):
    X = vectorizer.transform([input_text])
    pred = model.predict(X)
    return "Likely to Succeed" if pred[0] == 1 else "Unlikely to Succeed"

