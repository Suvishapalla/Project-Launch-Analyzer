import streamlit as st
from model.launch_predictor import predict_success
from utils.preprocess import preprocess_text

st.title("🚀 Project Launch Success Predictor")

title = st.text_input("Project Title")
description = st.text_area("Project Description")
tags = st.text_input("Tags (comma-separated)")

if st.button("Predict"):
    score = predict_success(title, description, tags)
    st.success(f"Predicted Success: {round(score * 100, 2)}%")
