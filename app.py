import streamlit as st
import joblib

# Load model & vectorizer
model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

st.title("Google Play Review Sentiment Classifier")

review = st.text_area("Enter a review:")

if st.button("Predict Sentiment"):
    if review.strip():
        vec = vectorizer.transform([review])
        prediction = model.predict(vec)[0]
        st.write(f"**Predicted Sentiment:** {prediction}")
    else:
        st.warning("Please enter a review.")
