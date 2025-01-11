# Text-Sentiment-Analysis
Overview
This project builds a Sentiment Analysis Model to classify text (e.g., movie reviews) as positive or negative. The task uses a dataset such as IMDB Reviews, preprocesses the text, converts it into numerical form, trains a machine learning model, and evaluates its performance.

Features
Text Preprocessing: Tokenization, stopword removal, lemmatization.
Feature Engineering: Text vectorization using TF-IDF.
Model Training: Logistic Regression classifier.
Model Evaluation: Metrics like accuracy, precision, recall, and F1-score.
Visualization: Graphs showcasing model performance and feature importance.
Prerequisites
Make sure the following are installed in your environment:

Python 3.7+
Libraries: pandas, numpy, nltk, sklearn, matplotlib, seaborn
Dataset
The dataset used is IMDB Reviews or any sentiment-labeled dataset.
Download the dataset from Kaggle IMDB Dataset.
Project Steps
1. Text Preprocessing
Tokenization: Split sentences into words.
Stopword Removal: Remove common words that add little meaning.
Lemmatization: Convert words to their base forms.
Tools: NLTK

2. Feature Engineering
TF-IDF Vectorization: Converts text into numerical format by representing words with their importance across documents.
Tools: Scikit-learn's TfidfVectorizer

3. Model Training
Classifier: Logistic Regression is used to predict the sentiment label (positive or negative).
Splitting Data: Train-test split (80%-20%).
Tools: Scikit-learn's LogisticRegression

4. Model Evaluation
Metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix.
Visualizations:
Accuracy and F1-score comparison.
Confusion Matrix heatmap.
Tools: Scikit-learn, Matplotlib, Seaborn
Results
Accuracy: 85% (Example result; varies depending on the dataset)
Precision/Recall/F1-score: Calculated and displayed after training.
Confusion Matrix: Visualized to show model predictions.
Visualizations
The project generates the following:

Accuracy Comparison: Bar chart comparing model accuracy.
Confusion Matrix Heatmap: Visualizes correct and incorrect predictions.
Acknowledgements
Dataset: Kaggle IMDB Dataset
Libraries: NLTK, Scikit-learn, Matplotlib, Seaborn

