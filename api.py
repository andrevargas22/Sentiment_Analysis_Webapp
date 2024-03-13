from fastapi import FastAPI
import mlflow.pyfunc
import numpy as np

import mlflow
import re
import requests
import os
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import pickle

nltk.download("stopwords")
stemmer = PorterStemmer()

with open("cache/sentiment_analysis/vocabulary.pkl", "rb") as f:
    vocabulary = pickle.load(f)
    
def review_to_words(review):
    text = BeautifulSoup(review, "html.parser").get_text() # Remove HTML tags
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()) # Convert to lower case
    words = text.split() # Split string into words
    words = [w for w in words if w not in stopwords.words("english")] # Remove stopwords
    words = [PorterStemmer().stem(w) for w in words] # Stemming    
    return words

def preprocess_input(review):
    # Preprocess the input review
    words = review_to_words(review)
    
    # Convert the input review into BoW features
    vectorizer = CountVectorizer(vocabulary=vocabulary)
    features = vectorizer.transform([' '.join(words)]).toarray()
    
    return features  

TRACKING_URI = "https://mlflow-server-wno7iop4fa-uc.a.run.app/"
mlflow.set_tracking_uri(TRACKING_URI)

# Fetch the model from the Model Registry
model_name = "sentiment_analysis"
stage = "staging"

model_uri = f"models:/{model_name}/{stage}"
loaded_model = mlflow.pyfunc.load_model(model_uri=model_uri)

# Create FastAPI app
app = FastAPI()

@app.post("/predict")
def predict_sentiment(text):
    """
    Predicts the sentiment of the input text.
    Args:
        input_text: The input text for sentiment classification.
    Returns:
        The predicted sentiment label (e.g., positive, negative).
    """
    
    features = preprocess_input(text)
    
    # Make prediction using the loaded model
    prediction = loaded_model.predict(features)
    
    # Convert prediction to string label
    sentiment_label = "positive" if prediction[0] == 1 else "negative"
    
    return {"sentiment": sentiment_label}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
