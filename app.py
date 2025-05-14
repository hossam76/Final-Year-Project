# === app.py: Deployment-Ready Flask App ===

import os
import pickle
import joblib
import re
import numpy as np
import pandas as pd
import json
import uuid
import datetime
import math
from collections import Counter
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from textblob import TextBlob
import base64
from io import BytesIO

# Flask setup
from flask import Flask, request, jsonify, render_template, redirect, url_for, session, send_file
from flask_caching import Cache

app = Flask(__name__, static_folder='static')
app.secret_key = os.urandom(24)
app.jinja_env.globals.update(min=min, max=max)

cache = Cache(app, config={
    "DEBUG": True,
    "CACHE_TYPE": "SimpleCache",
    "CACHE_DEFAULT_TIMEOUT": 300
})

os.makedirs('analysis_data', exist_ok=True)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# === Conditional Imports ===
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    HAS_SPACY = True
except:
    HAS_SPACY = False

try:
    import lime
    from lime.lime_text import LimeTextExplainer
    explainer = LimeTextExplainer(class_names=['fake', 'real'])
    HAS_LIME = True
except:
    HAS_LIME = False
    explainer = None

# === Utilities ===
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    tokens = word_tokenize(text)
    return ' '.join([lemmatizer.lemmatize(word) for word in tokens if word not in stop_words])

def predict_with_models(text):
    processed = preprocess_text(text)
    vectorizer = joblib.load('vectorizer.pkl')
    nb_model = joblib.load('naive_bayes_model.pkl')
    rf_model = joblib.load('random_forest_model.pkl')
    vect = vectorizer.transform([processed])

    result = {
        'naive_bayes': {
            'prediction': 'real' if nb_model.predict(vect)[0] == 1 else 'fake',
            'confidence': float(nb_model.predict_proba(vect)[0].max())
        },
        'random_forest': {
            'prediction': 'real' if rf_model.predict(vect)[0] == 1 else 'fake',
            'confidence': float(rf_model.predict_proba(vect)[0].max())
        }
    }

    if HAS_TENSORFLOW:
        try:
            tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
            lstm_model = load_model('lstm_model.h5')
            padded = pad_sequences(tokenizer.texts_to_sequences([processed]), maxlen=100)
            pred = lstm_model.predict(padded)[0][0]
            result['lstm'] = {
                'prediction': 'real' if pred > 0.5 else 'fake',
                'confidence': float(pred if pred > 0.5 else 1 - pred)
            }
        except:
            pass

    votes = [1 if result[m]['prediction'] == 'real' else 0 for m in result]
    result['ensemble'] = {
        'prediction': 'real' if sum(votes) > len(votes)//2 else 'fake',
        'confidence': round(sum(result[m]['confidence'] for m in result if m != 'ensemble') / len(votes), 2)
    }
    return result

# === Routes ===
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '')
        model_choice = data.get('model', 'naive_bayes')
        if not text:
            return jsonify({'error': 'Text is required'}), 400

        processed = preprocess_text(text)

        if model_choice == 'lstm' and HAS_TENSORFLOW:
            tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
            lstm_model = load_model('lstm_model.h5')
            padded = pad_sequences(tokenizer.texts_to_sequences([processed]), maxlen=100)
            prob = lstm_model.predict(padded)[0][0]
            return jsonify({
                'prediction': 'real' if prob > 0.5 else 'fake',
                'confidence': float(prob)
            })

        vectorizer = joblib.load('vectorizer.pkl')
        model_file = 'random_forest_model.pkl' if model_choice == 'random_forest' else 'naive_bayes_model.pkl'
        model = joblib.load(model_file)
        vect = vectorizer.transform([processed])
        pred = model.predict(vect)[0]
        prob = model.predict_proba(vect)[0].max()

        return jsonify({
            'prediction': 'real' if pred == 1 else 'fake',
            'confidence': float(prob)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_ensemble', methods=['POST'])
def predict_ensemble():
    try:
        data = request.get_json()
        text = data.get('text', '')
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        result = predict_with_models(text)
        return jsonify(result['ensemble'])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/hello')
def hello():
    return "Hello, world!"

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

