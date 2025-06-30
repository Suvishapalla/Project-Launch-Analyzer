from model.launch_predictor import load_model_and_vectorizer, predict
from utils.preprocess import preprocess_text
import json

if __name__ == "__main__":
    model, vectorizer = load_model_and_vectorizer()
    
    with open("data/test_inputs.json", "r") as f:
        ideas = json.load(f)

    for idea in ideas:
        clean_text = preprocess_text(idea["description"])
        result = predict(model, vectorizer, clean_text)
        print(f"'{idea['title']}' prediction: {result}")
