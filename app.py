import pickle
from flask import Flask, render_template, request, jsonify
import re
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

try:
    # Download required NLTK data
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
except Exception as e:
    print(f"Warning: Could not download NLTK data: {e}")

app = Flask(__name__)

# Load pickled files & data
try:
    with open('count_vectorizer.pkl', 'rb') as f:
        cv = pickle.load(f)
    with open('tfidf_transformer.pkl', 'rb') as f:
        tfidf_transformer = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
except FileNotFoundError as e:
    print(f"Error loading pickle files: {e}")
    cv = None
    tfidf_transformer = None
    feature_names = None

# Cleaning data:
stop_words = set(stopwords.words('english'))
new_stop_words = ["fig","figure","image","sample","using",
             "show", "result", "large",
             "also", "one", "two", "three",
             "four", "five", "seven","eight","nine"]
stop_words = list(stop_words.union(new_stop_words))

def preprocess_text(txt):
    try:
        # Lower case
        txt = txt.lower()
        # Remove HTML tags
        txt = re.sub(r"<.*?>", " ", txt)
        # Remove special characters and digits
        txt = re.sub(r"[^a-zA-Z]", " ", txt)
        # tokenization
        txt = nltk.word_tokenize(txt)
        # Remove stopwords
        txt = [word for word in txt if word not in stop_words]
        # Remove words less than three letters
        txt = [word for word in txt if len(word) >= 3]
        # Lemmatize
        lmtr = WordNetLemmatizer()
        txt = [lmtr.lemmatize(word) for word in txt]
        return " ".join(txt)
    except Exception as e:
        print(f"Error in preprocess_text: {e}")
        return txt

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    sorted_items = sorted_items[:topn]
    score_vals = []
    feature_vals = []
    
    for idx, score in sorted_items:
        fname = feature_names[idx]
        score_vals.append(round(score, 3))
        feature_vals.append(fname)

    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]
    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/extract_keywords', methods=['POST'])
def extract_keywords():
    try:
        document = request.files.get('file')
        if not document or document.filename == '':
            return render_template('index.html', error='No document selected')

        text = document.read().decode('utf-8', errors='ignore')
        preprocessed_text = preprocess_text(text)
        tf_idf_vector = tfidf_transformer.transform(cv.transform([preprocessed_text]))
        sorted_items = sort_coo(tf_idf_vector.tocoo())
        keywords = extract_topn_from_vector(feature_names, sorted_items, 20)
        return render_template('keywords.html', keywords=keywords)
    except Exception as e:
        return render_template('index.html', error=f'Error: {str(e)}'), 500

@app.route('/search_keywords', methods=['POST'])
def search_keywords():
    try:
        search_query = request.form.get('search', '')
        if search_query:
            keywords = []
            for keyword in feature_names:
                if search_query.lower() in keyword.lower():
                    keywords.append(keyword)
                    if len(keywords) == 20:
                        break
            return render_template('keywordslist.html', keywords=keywords)
        return render_template('index.html', error='No search query provided')
    except Exception as e:
        return render_template('index.html', error=f'Error: {str(e)}'), 500

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True)