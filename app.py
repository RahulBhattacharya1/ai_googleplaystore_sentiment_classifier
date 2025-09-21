import streamlit as st
import joblib

@st.cache_resource(show_spinner=False)
def load_pipeline():
    # Single portable artifact – no separate vectorizer/model
    return joblib.load("models/sentiment_pipeline.joblib")

st.title("Google Play Review Sentiment Classifier")

try:
    pipe = load_pipeline()
except Exception as e:
    st.error(
        "Failed to load the model pipeline. "
        "Ensure 'models/sentiment_pipeline.joblib' exists and matches the library versions."
    )
    st.exception(e)
    st.stop()

txt = st.text_area("Enter a review to analyze:")
if st.button("Predict"):
    review = txt.strip()
    if not review:
        st.warning("Please enter a review.")
    else:
        pred = pipe.predict([review])[0]
        proba = None
        # If classifier supports predict_proba, show confidence
        if hasattr(pipe, "predict_proba"):
            import numpy as np
            proba = pipe.predict_proba([review])[0]
            conf = float(np.max(proba))
            st.write(f"**Predicted Sentiment:** {pred}")
            st.write(f"Confidence: {conf:.2f}")
        else:
            st.write(f"**Predicted Sentiment:** {pred}")
