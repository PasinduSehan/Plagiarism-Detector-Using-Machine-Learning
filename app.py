# from flask import Flask, render_template, request
# import pickle

# app = Flask(__name__)

# model = pickle.load(open('C:\Users\User\Desktop\Project_hello_hello\ML_PROJECT\plagrisum_detector_ml\Plagiarism_checker_ML\model.pkl', 'rb'))
# tfidf_vectorizer = pickle.load(open('C:\Users\User\Desktop\Project_hello_hello\ML_PROJECT\plagrisum_detector_ml\Plagiarism_checker_ML\tfidf_vectorizer.pkl', 'rb'))

# def detect(input_text):
#     vectorized_text = tfidf_vectorizer.transform([input_text])
#     result = model.predict(vectorized_text)
#     return "Plagiarism Detected" if result[0] == 1 else "No Plagiarism Detected"

# @app.route('/')
# def home():
#     return render_template('C:\Users\User\Desktop\Project_hello_hello\ML_PROJECT\plagrisum_detector_ml\Plagiarism_checker_ML\index.html\index.html')

# @app.route('/detect', methods=['POST'])
# def detect_plagiarism():
#     input_text = request.form['text']
#     detection_result = detect(input_text)
#     return render_template('C:\Users\User\Desktop\Project_hello_hello\ML_PROJECT\plagrisum_detector_ml\Plagiarism_checker_ML\index.html\index.html', result=detection_result)

# if __name__ == "__main__":
#     app.run(debug=True)
    



from flask import Flask, render_template, request
import pickle
import string
from nltk.corpus import stopwords

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# Preprocessing function (same as in the notebook)
def preprocess_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Convert to lowercase
    text = text.lower()
    # Remove stop words
    stop_words = set(stopwords.words("english"))
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text

def detect(input_text):
    # Preprocess the input text
    processed_text = preprocess_text(input_text)
    # Vectorize the text
    vectorized_text = tfidf_vectorizer.transform([processed_text])
    # Predict
    result = model.predict(vectorized_text)
    return "Plagiarism Detected" if result[0] == 1 else "No Plagiarism Detected"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_plagiarism():
    input_text = request.form['text']
    detection_result = detect(input_text)
    return render_template('index.html', result=detection_result)

if __name__ == "__main__":
    app.run(debug=True)
    