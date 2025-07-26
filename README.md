"# Plagiarism-Detector-Using-Machine-Learning" 
Plagiarism Detector Using Machine Learning
Introduction
This project implements a plagiarism detection system using machine learning. It leverages TF-IDF vectorization and a Support Vector Machine (SVM) model to classify text as plagiarized or original. The system is deployed as a Flask web application, providing a user-friendly interface to input text and receive plagiarism detection results.
Project Structure
Plagiarism_checker_ML/
├── app.py                                    # Flask application for the web interface ==>
├── templates/
│   └── index.html                           # HTML template for the web interface==>
├── model.pkl                                # Trained SVM model (serialized)==>
├── tfidf_vectorizer.pkl                     # TF-IDF vectorizer (serialized)==>
├── dataset.csv                              # Dataset for training the model==>
├── README.md                                # Project documentation==>
├── Building Plagiarism checker using Machine Learning.ipynb  # Jupyter notebook for model training==>

Requirements

Python: 3.9.12 or compatible (tested with 3.12)
Dependencies:
flask==3.1.1
pandas
scikit-learn==1.3.2
nltk
joblib
jupyter (for running the notebook)


Dataset: dataset.csv with columns: Unnamed: 0, source_text, plagiarized_text, label (0 for non-plagiarized, 1 for plagiarized)

