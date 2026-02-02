import pickle
import os
from flask import Flask, render_template, request, jsonify
import re
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

try:
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
except Exception as e:
    print(f"NLTK download warning: {e}")

# Get correct template path
template_dir = os.path.join(os.path.dirname(__file__), '../templates')
if not os.path.exists(template_dir):
    os.makedirs(template_dir, exist_ok=True)

app = Flask(__name__, template_folder=template_dir, static_folder=os.path.join(os.path.dirname(__file__), '../static'))

# Load pickle files
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cv = None
tfidf_transformer = None
feature_names = None

try:
    cv = pickle.load(open(os.path.join(base_dir, 'count_vectorizer.pkl'), 'rb'))
    tfidf_transformer = pickle.load(open(os.path.join(base_dir, 'tfidf_transformer.pkl'), 'rb'))
    feature_names = pickle.load(open(os.path.join(base_dir, 'feature_names.pkl'), 'rb'))
except FileNotFoundError:
    print("Warning: Pickle files not found")

stop_words = set(stopwords.words('english'))
new_stop_words = ["fig","figure","image","sample","using","show","result","large","also","one","two","three","four","five","seven","eight","nine"]
stop_words = list(stop_words.union(new_stop_words))

def preprocess_text(txt):
    try:
        txt = txt.lower()
        txt = re.sub(r"<.*?>", " ", txt)
        txt = re.sub(r"[^a-zA-Z]", " ", txt)
        txt = nltk.word_tokenize(txt)
        txt = [word for word in txt if word not in stop_words and len(word) >= 3]
        lmtr = WordNetLemmatizer()
        txt = [lmtr.lemmatize(word) for word in txt]
        return " ".join(txt)
    except Exception as e:
        return txt

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    sorted_items = sorted_items[:topn]
    results = {}
    for idx, score in sorted_items:
        results[feature_names[idx]] = round(score, 3)
    return results

@app.route('/')
def index():
    return jsonify({"message": "Keyword Extraction API", "endpoints": ["/health", "/extract_keywords"]})

@app.route('/extract_keywords', methods=['POST'])
def extract_keywords():
    try:
        if not all([cv, tfidf_transformer, feature_names]):
            return jsonify({"error": "Model files not loaded"}), 503
        
        document = request.files.get('file')
        if not document:
            return jsonify({"error": "No file uploaded"}), 400
        
        text = document.read().decode('utf-8', errors='ignore')
        preprocessed = preprocess_text(text)
        
        if not preprocessed.strip():
            return jsonify({"error": "No valid text extracted"}), 400
        
        tf_idf = tfidf_transformer.transform(cv.transform([preprocessed]))
        keywords = extract_topn_from_vector(feature_names, sort_coo(tf_idf.tocoo()), 20)
        return jsonify(keywords)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health')
def health():
    return jsonify({"status": "ok", "models_loaded": all([cv, tfidf_transformer, feature_names])})

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Server error"}), 500