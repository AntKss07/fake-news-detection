# app.py

import streamlit as st
import sys
st.write(f"Python executable: {sys.executable}")
st.write(f"Python path: {sys.path}")
import joblib
import re
import string
from nltk.corpus import stopwords
import os

# Define the path where models are saved
MODEL_PATH = 'models/'

# Ensure NLTK resources are available for the Streamlit app
try:
    stopwords.words('english')
except LookupError:
    import nltk
    nltk.download('stopwords')

# Preprocessing function (same as in fake_news_classifier.py)
def preprocess_text(text):
    text = text.lower()  # Lowercasing
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)  # Remove punctuation
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    return text

# Function to load models (same as in fake_news_classifier.py)
@st.cache_resource
def load_models():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model directory '{MODEL_PATH}' not found. Please run 'python fake_news_classifier.py' to train and save the models first.")
        return None, None, None

    try:
        vectorizer = joblib.load(os.path.join(MODEL_PATH, 'vectorizer.joblib'))
        mnb_model = joblib.load(os.path.join(MODEL_PATH, 'mnb_model.joblib'))
        lr_model = joblib.load(os.path.join(MODEL_PATH, 'lr_model.joblib'))
        return vectorizer, mnb_model, lr_model
    except FileNotFoundError:
        st.error(f"One or more model files not found in '{MODEL_PATH}'. Please run 'python fake_news_classifier.py' to train and save the models first.")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

# Load models and vectorizer
vectorizer, mnb_model, lr_model = load_models()

# Streamlit UI
st.set_page_config(page_title="Fake News Classifier", layout="centered")

st.title("📰 Fake News Classifier")
st.markdown("""
    Enter a news article in the text area below and click "Predict" 
    to classify it as either Real or Fake using two different machine learning models.
""")

news_input = st.text_area("Enter News Article Here:", height=250, 
                          placeholder="Type or paste the news article content...")

if st.button("Predict"):
    if news_input:
        if vectorizer is None or mnb_model is None or lr_model is None:
            st.warning("Models are not loaded. Please ensure 'fake_news_classifier.py' was run successfully to train and save models.")
        else:
            st.subheader("Prediction Results:")
            
            cleaned_text = preprocess_text(news_input)
            vectorized_text = vectorizer.transform([cleaned_text])

            # Multinomial Naive Bayes Prediction
            mnb_prediction = mnb_model.predict(vectorized_text)[0]
            mnb_result = "Fake" if mnb_prediction == 1 else "Real"
            st.info(f"**Multinomial Naive Bayes:** {mnb_result} News")

            # Logistic Regression Prediction
            lr_prediction = lr_model.predict(vectorized_text)[0]
            lr_result = "Fake" if lr_prediction == 1 else "Real"
            st.success(f"**Logistic Regression:** {lr_result} News")

            st.markdown("---")
            st.caption("Note: These predictions are based on models trained on a small dummy dataset. Real-world performance may vary significantly.")
    else:
        st.warning("Please enter some news text to get a prediction.")

st.markdown("---")
st.markdown("### How it works:")
st.markdown("""
1.  **Text Preprocessing:** The input text is converted to lowercase, punctuation is removed, and common English stop words (like 'the', 'is') are removed.
2.  **Feature Extraction:** The cleaned text is transformed into a numerical representation using a Bag-of-Words (BoW) approach. This counts the occurrences of each word.
3.  **Model Prediction:** Two trained machine learning models (Multinomial Naive Bayes and Logistic Regression) then analyze these numerical features to classify the news as Real or Fake.
""")
