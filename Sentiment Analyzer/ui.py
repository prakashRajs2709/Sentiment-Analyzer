from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from string import punctuation
import re
from nltk.corpus import stopwords

nltk.download('stopwords')

set(stopwords.words('english'))
import streamlit as st
import pickle

def load_model():
    with open("Sentiment Analysis/Sentiment_Analysis2.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# Function to predict sentiment
def process_text(text):
    stop_words = stopwords.words('english')
    text1 = text.lower()
    text_final = ''.join(c for c in text1 if not c.isdigit())
    processed_doc1 = ' '.join([word for word in text_final.split() if word not in stop_words])
    return processed_doc1

def analyze_sentiment(text):
    sa = SentimentIntensityAnalyzer()
    scores = sa.polarity_scores(text=text)
    compound = round((1 + scores['compound'])/2, 2)
    return compound,scores



st.set_page_config(page_title="Deep Learning for Sentiment Analysis", page_icon="😁")
st.title("Sentiment Analysis App")


# Input text
user_input = st.text_area("Enter text here:")

if st.button("Analyze"):
    if user_input:
        processed_text = process_text(user_input)
        compound,scores = analyze_sentiment(processed_text)
        st.write('Original Text:', user_input)
        st.write('Positive:', scores['pos'])
        st.write('Negative:', scores['neg'])
        st.write('Neutral:', scores['neu'])
        st.write('Compound Score:', compound)
    else:
        st.write("Please enter some text to analyze.")
