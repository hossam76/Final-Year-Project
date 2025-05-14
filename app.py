import os
import pickle
import joblib
import re
import numpy as np
import pandas as pd
import json
from collections import Counter
from urllib.parse import urlparse
import base64
from io import BytesIO

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

# TensorFlow for LSTM
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

# LIME for explainability
try:
    import lime
    from lime.lime_text import LimeTextExplainer
    HAS_LIME = True
except ImportError:
    HAS_LIME = False

# Flask
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# NLTK setup
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load models on demand
vectorizer = joblib.load('vectorizer.pkl')
nb_model = joblib.load('naive_bayes_model.pkl')
rf_model = joblib.load('random_forest_model.pkl')
lstm_model = load_model('lstm_model.h5') if HAS_TENSORFLOW else None
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Text preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    return ' '.join([lemmatizer.lemmatize(t) for t in tokens if t not in stop_words])

# Word cloud
def generate_wordcloud(text):
    words = [word.lower() for word in word_tokenize(text)
             if word.isalpha() and word.lower() not in stop_words]
    text_joined = ' '.join(words)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_joined)
    img = BytesIO()
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    return f"data:image/png;base64,{base64.b64encode(img.getvalue()).decode('utf-8')}"

# LIME explanation
def explain_prediction(text):
    if not HAS_LIME:
        return {'error': 'LIME not available'}
    explainer = LimeTextExplainer(class_names=['fake', 'real'])
    def predictor(texts):
        preprocessed = [preprocess_text(t) for t in texts]
        vect = vectorizer.transform(preprocessed)
        return nb_model.predict_proba(vect)
    explanation = explainer.explain_instance(text, predictor, num_features=10)
    return {
        'explanation': explanation.as_list(),
        'html': explanation.as_html()
    }

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
            return jsonify({'error': 'No text provided'}), 400

        processed = preprocess_text(text)
        result = {}

        if model_choice == 'lstm' and HAS_TENSORFLOW:
            seq = tokenizer.texts_to_sequences([processed])
            pad = pad_sequences(seq, maxlen=100)
            prob = lstm_model.predict(pad)[0][0]
            result['prediction'] = 'real' if prob > 0.5 else 'fake'
            result['confidence'] = float(prob)
        else:
            model = rf_model if model_choice == 'random_forest' else nb_model
            vect = vectorizer.transform([processed])
            pred = model.predict(vect)[0]
            prob = model.predict_proba(vect)[0].max()
            result['prediction'] = 'real' if pred == 1 else 'fake'
            result['confidence'] = float(prob)

        result['wordcloud'] = generate_wordcloud(text)
        result['lime'] = explain_prediction(text)

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

