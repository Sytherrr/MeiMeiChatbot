import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load the dataset
with open('syn_ant_sent_sim.json', 'r') as f:
    data = json.load(f)

# Prepare the data
words_syn = []
synonyms = []
words_ant = []
antonyms = []

for item in data:
    if 'syn_list' in item and len(item['syn_list']) > 0:  # Check if 'syn_list' exists and is not empty
        words_syn.append(item['word'])
        synonyms.append(item['syn_list'][0]['synonym'])  # This is a simplification, you might want to handle synonyms and antonyms differently
    if 'ant_list' in item and len(item['ant_list']) > 0:  # Check if 'ant_list' exists and is not empty
        words_ant.append(item['word'])
        antonyms.append(item['ant_list'][0]['antonym'])  # This is a simplification, you might want to handle synonyms and antonyms differently

# Split the data into training and test sets
X_train_syn, X_test_syn, y_train_syn, y_test_syn = train_test_split(words_syn, synonyms, test_size=0.2, random_state=42)
X_train_ant, X_test_ant, y_train_ant, y_test_ant = train_test_split(words_ant, antonyms, test_size=0.2, random_state=42)

# Create a pipeline that first transforms the data using TfidfVectorizer, then trains a model using LinearSVC
pipeline_syn = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LinearSVC()),
])

pipeline_ant = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LinearSVC()),
])

# Train the model for synonyms
pipeline_syn.fit(X_train_syn, y_train_syn)

# Train the model for antonyms
pipeline_ant.fit(X_train_ant, y_train_ant)

# Save the models
pipeline_syn.fit(X_train_syn, y_train_syn)
pipeline_ant.fit(X_train_ant, y_train_ant)

# Save the trained models
joblib.dump(pipeline_syn, 'synonym_model.pkl')
joblib.dump(pipeline_ant, 'antonym_model.pkl')

# Test the model for synonyms
predictions_syn = pipeline_syn.predict(X_test_syn)

# Test the model for antonyms
predictions_ant = pipeline_ant.predict(X_test_ant)

# Print a classification report for synonyms
print("Classification report for synonyms:")
print(classification_report(y_test_syn, predictions_syn))

# Print a classification report for antonyms
print("Classification report for antonyms:")
print(classification_report(y_test_ant, predictions_ant))
