import json
import joblib
import string
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

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
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

    preprocessed_text = ' '.join(lemmatized_tokens)
    return preprocessed_text

# Preprocess the entire dataset
X_preprocessed_syn = [preprocess_text(text) for text in words_syn]
X_preprocessed_ant = [preprocess_text(text) for text in words_ant]

# Display original and preprocessed text for synonyms
print("Synonyms:")
for i in range(len(words_syn)):
    print("Original Text:", words_syn[i])
    print("Preprocessed Text:", X_preprocessed_syn[i])
    print()

# Display original and preprocessed text for antonyms
print("Antonyms:")
for i in range(len(words_ant)):
    print("Original Text:", words_ant[i])
    print("Preprocessed Text:", X_preprocessed_ant[i])
    print()

# Create a pipeline TfidfVectorizer, then trains a model using LinearSVC
pipeline_syn = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LinearSVC(dual=False))
])

pipeline_ant = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LinearSVC(dual=False))
])

# Train the model
pipeline_syn.fit(X_train_syn, y_train_syn)
pipeline_ant.fit(X_train_ant, y_train_ant)

# Test the model for synonyms
predictions_syn = pipeline_syn.predict(X_test_syn)

# Test the model for antonyms
predictions_ant = pipeline_ant.predict(X_test_ant)

# Calculate accuracy for synonyms
synonyms_accuracy = accuracy_score(y_test_syn, predictions_syn)

# Calculate accuracy for antonyms
antonyms_accuracy = accuracy_score(y_test_ant, predictions_ant)

# Print a classification report for synonyms
print("Classification report for synonyms:")
print(classification_report(y_test_syn, predictions_syn))

# Print a classification report for antonyms
print("Classification report for antonyms:")
print(classification_report(y_test_ant, predictions_ant))

# Visualize confusion matrix for synonyms
synonyms_cm = confusion_matrix(y_test_syn, predictions_syn)
plt.figure(figsize=(8, 6))
sns.heatmap(synonyms_cm, annot=True, fmt='d', cmap='Blues', xticklabels=pipeline_syn.classes_, yticklabels=pipeline_syn.classes_)
plt.title('Confusion Matrix for Synonyms')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Visualize confusion matrix for antonyms
antonyms_cm = confusion_matrix(y_test_ant, predictions_ant)
plt.figure(figsize=(8, 6))
sns.heatmap(antonyms_cm, annot=True, fmt='d', cmap='Blues', xticklabels=pipeline_ant.classes_, yticklabels=pipeline_ant.classes_)
plt.title('Confusion Matrix for Antonyms')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
