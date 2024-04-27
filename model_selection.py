import json
import string
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline

# Load dataset
with open('syn_ant_sent_sim.json', 'r') as f:
    data = json.load(f)

# Prepare the data
words_syn = []
synonyms = []
words_ant = []
antonyms = []

for item in data:
    if 'syn_list' in item and len(item['syn_list']) > 0:
        words_syn.append(item['word'])
        synonyms.append(item['syn_list'][0]['synonym'])
    if 'ant_list' in item and len(item['ant_list']) > 0:
        words_ant.append(item['word'])
        antonyms.append(item['ant_list'][0]['antonym'])

# Split the data into training and test sets
X_train_syn, X_test_syn, y_train_syn, y_test_syn = train_test_split(words_syn, synonyms, test_size=0.2, random_state=42)
X_train_ant, X_test_ant, y_train_ant, y_test_ant = train_test_split(words_ant, antonyms, test_size=0.2, random_state=42)

# Text Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

    preprocessed_text = ' '.join(lemmatized_tokens)
    return preprocessed_text

# Preprocess the entire dataset
X_preprocessed_syn = [preprocess_text(text) for text in words_syn]
X_preprocessed_ant = [preprocess_text(text) for text in words_ant]

# Function to evaluate model performance
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy:", accuracy)
    print("Classification report:")
    print(classification_report(y_test, predictions))
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# List of classifiers to experiment with
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

# Train and evaluate models for synonyms
print("Synonyms:")
for name, classifier in classifiers.items():
    print("\nTraining and evaluating", name)
    pipeline_syn = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', classifier)
    ])
    pipeline_syn.fit(X_train_syn, y_train_syn)
    print("Evaluation results for", name)
    evaluate_model(pipeline_syn, X_test_syn, y_test_syn)

# Train and evaluate models for antonyms
print("Antonyms:")
for name, classifier in classifiers.items():
    print("\nTraining and evaluating", name)
    pipeline_ant = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', classifier)
    ])
    pipeline_ant.fit(X_train_ant, y_train_ant)
    print("Evaluation results for", name)
    evaluate_model(pipeline_ant, X_test_ant, y_test_ant)
