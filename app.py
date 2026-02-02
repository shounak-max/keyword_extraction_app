import pickle
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
    print(f"Warning: {e}")

app = Flask(__name__)

try:
    with open('count_vectorizer.pkl', 'rb') as f:
        cv = pickle.load(f)
    with open('tfidf_transformer.pkl', 'rb') as f:
        tfidf_transformer = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
except:
    cv = tfidf_transformer = feature_names = None

stop_words = set(stopwords.words('english'))
new_stop_words = ["fig","figure","image","sample","using","show", "result", "large","also", "one", "two", "three","four", "five", "seven","eight","nine"]
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
    return render_template('index.html')

@app.route('/extract_keywords', methods=['POST'])
def extract_keywords():
    try:
        document = request.files.get('file')
        if not document:
            return jsonify({"error": "No file"}), 400
        text = document.read().decode('utf-8', errors='ignore')
        preprocessed = preprocess_text(text)
        tf_idf = tfidf_transformer.transform(cv.transform([preprocessed]))
        keywords = extract_topn_from_vector(feature_names, sort_coo(tf_idf.tocoo()), 20)
        return jsonify(keywords)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Server error"}), 500

if __name__ == '__main__':
    app.run(debug=True)