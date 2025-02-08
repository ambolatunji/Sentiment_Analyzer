import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from wordcloud import WordCloud
import emoji
import torch
import os
from pathlib import Path
import logging
import traceback
import re
from collections import defaultdict
from docx import Document
import PyPDF2
import io
import base64
import tempfile
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from flair.models import TextClassifier
from flair.data import Sentence

# Download necessary NLTK data
nltk.download('vader_lexicon')

# Initialize NLTK and Flair sentiment analyzers
sia = SentimentIntensityAnalyzer()
flair_classifier = TextClassifier.load('en-sentiment')

def setup_logging():
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)

def initialize_model(model_name, logger, cache_path):
    try:
        cache_dir = Path(cache_path)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Using cache directory: {cache_dir}")
        logger.info(f"Attempting to load model: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir)
        
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        
        logger.info("Model loaded successfully")
        return sentiment_pipeline
    
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}\n{traceback.format_exc()}")
        st.error(f"An error occurred while loading the model: {str(e)}")
        return None

def analyze_with_nltk(text):
    return sia.polarity_scores(text)

def analyze_with_flair(text):
    sentence = Sentence(text)
    flair_classifier.predict(sentence)
    return sentence.labels[0].to_dict()

def split_text(text):
    return re.split(r'[.,!?;:]', text)

def detect_sentiments(sentences, sentiment_pipeline, method="transformer"):
    sentiments = []
    confidence_scores = []
    report = ""
    
    for text in sentences:
        if method == "transformer":
            result = sentiment_pipeline(text)[0]
            sentiment = result['label']
            confidence = result['score']
        elif method == "nltk":
            result = analyze_with_nltk(text)
            sentiment = max(result, key=result.get)  # Pick strongest sentiment
            confidence = result[sentiment]
        elif method == "flair":
            result = analyze_with_flair(text)
            sentiment = result['value']
            confidence = result['confidence']
        
        sentiments.append(sentiment)
        confidence_scores.append(confidence)
        
        report += f"**Segment:** {text.strip()}\n"
        report += f"Sentiment: {sentiment} (Confidence: {confidence:.2f})\n\n"
        report += "---\n"
    
    df = pd.DataFrame({"Sentiment": sentiments, "Confidence": confidence_scores})
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="Sentiment", palette="coolwarm", ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    return report

def run():
    logger = setup_logging()
    st.header("📖 Text-Based Sentiment Analysis")
    
    model_options = {
        "DistilBERT (Fast)": "distilbert-base-uncased-finetuned-sst-2-english",
        "BERT": "nlptown/bert-base-multilingual-uncased-sentiment",
        "RoBERTa": "cardiffnlp/twitter-roberta-base-sentiment"
    }
    
    analysis_methods = {"Transformer Model": "transformer", "NLTK VADER": "nltk", "Flair": "flair"}
    selected_method = st.selectbox("Choose analysis method:", list(analysis_methods.keys()))
    
    sentiment_pipeline = None
    if selected_method == "Transformer Model":
        selected_model = st.selectbox("Choose a sentiment analysis model:", list(model_options.keys()))
        sentiment_pipeline = initialize_model(model_options[selected_model], logger, os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
    
    user_input = st.text_area("Enter text for sentiment analysis:")
    
    if st.button("Analyze"):
        if user_input:
            sentences = [s.strip() for s in split_text(user_input) if s.strip()]
            report = detect_sentiments(sentences, sentiment_pipeline, method=analysis_methods[selected_method])
            
            pdf_data = generate_pdf(report)
            st.download_button("Download Analysis as PDF", pdf_data, "analysis.pdf", "application/pdf")

if __name__ == "__main__":
    run()
