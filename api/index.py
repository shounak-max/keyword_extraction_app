import pickle
import os
from flask import Flask, jsonify, request
import re
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

# Download NLTK data
for resource in ['wordnet', 'averaged_perceptron_tagger', 'omw-1.4', 'stopwords', 'punkt']:
    try:
        nltk.download(resource, quiet=True)
    except:
        pass

app = Flask(__name__)

# Load models - check multiple paths
cv = None
tfidf_transformer = None
feature_names = None

paths_to_try = [
    '/var/task/',  # Vercel path
    '/tmp/',
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  # Parent directory
]

for base_path in paths_to_try:
    try:
        cv_path = os.path.join(base_path, 'count_vectorizer.pkl')
        tfidf_path = os.path.join(base_path, 'tfidf_transformer.pkl')
        features_path = os.path.join(base_path, 'feature_names.pkl')
        
        if os.path.exists(cv_path):
            cv = pickle.load(open(cv_path, 'rb'))
            tfidf_transformer = pickle.load(open(tfidf_path, 'rb'))
            feature_names = pickle.load(open(features_path, 'rb'))
            print(f"✓ Models loaded from {base_path}")
            break
    except Exception as e:
        print(f"✗ Failed to load from {base_path}: {e}")
        continue

stop_words = set(stopwords.words('english'))
new_stop_words = ["fig", "figure", "image", "sample", "using", "show", "result", "large", "also", "one", "two", "three", "four", "five", "seven", "eight", "nine"]
stop_words.update(new_stop_words)

def preprocess_text(txt):
    try:
        txt = txt.lower()
        txt = re.sub(r"<.*?>", " ", txt)
        txt = re.sub(r"[^a-zA-Z\s]", " ", txt)
        tokens = nltk.word_tokenize(txt)
        tokens = [word for word in tokens if word not in stop_words and len(word) >= 3]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return " ".join(tokens)
    except Exception as e:
        print(f"Preprocess error: {e}")
        return ""

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn(feature_names, sorted_items, topn=10):
    results = {}
    for idx, score in sorted_items[:topn]:
        results[feature_names[idx]] = round(float(score), 3)
    return results

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "status": "running",
        "app": "Keyword Extraction API",
        "endpoints": ["/health", "/extract_keywords"],
        "models_loaded": bool(all([cv, tfidf_transformer, feature_names]))
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "models_loaded": bool(all([cv, tfidf_transformer, feature_names]))
    })

@app.route('/extract_keywords', methods=['POST'])
def extract_keywords():
    try:
        if not all([cv, tfidf_transformer, feature_names]):
            return jsonify({"error": "Models not loaded. Upload pickle files to Vercel."}), 503
        
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400
        
        text = file.read().decode('utf-8', errors='ignore')
        preprocessed = preprocess_text(text)
        
        if not preprocessed.strip():
            return jsonify({"error": "No valid content"}), 400
        
        tfidf_matrix = tfidf_transformer.transform(cv.transform([preprocessed]))
        keywords = extract_topn(feature_names, sort_coo(tfidf_matrix.tocoo()), 20)
        
        return jsonify(keywords)
    except Exception as e:
        return jsonify({"error": str(e), "type": type(e).__name__}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500