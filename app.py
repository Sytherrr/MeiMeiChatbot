from flask import Flask, render_template, request, jsonify
import openai
import spacy
import nltk
from nltk.corpus import wordnet
import joblib
import numpy as np

app = Flask(__name__)

# OpenAI API key
api_key = "..."
openai.api_key = api_key

# Load spaCy's English language model
nlp = spacy.load('en_core_web_md')

# Load the trained synonym and antonym models
synonym_model = joblib.load('synonym_model.pkl')
antonym_model = joblib.load('antonym_model.pkl')

# Function word embeddings
def get_word_embedding(word):
    return nlp(word).vector

# Finding synonyms using WordNet
def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    return synonyms

# Finding antonyms using WordNet
def get_antonyms(word):
    antonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if lemma.antonyms():
                antonyms.append(lemma.antonyms()[0].name())
    return antonyms

# Command type validation
def validate_command_type(command_type):
    if command_type not in ["grammar", "translate", "antonym-synonym"]:
        raise ValueError("Invalid command type")

# Error handling
def handle_error(error):
    return f"An error occurred: {error}"

# Chatbot function
def chat_with_bot(prompt, command_type):
    if command_type == "antonym-synonym":
        if not prompt:  # Check if user provided input
            return "What sentence or word do you want to inquire about?"
        else:
            try:
                word = prompt.lower()  # Convert to lowercase
                synonyms_list = get_synonyms(word)
                antonyms_list = get_antonyms(word)
                synonyms_str = ', '.join(synonyms_list)
                antonyms_str = ', '.join(antonyms_list)
                return f"Synonyms for '{word}' is {synonyms_str}\n\nAntonyms for '{word}' is {antonyms_str}"
            except Exception as e:
                return f"An error occurred: {str(e)}"
    elif command_type == "translate":
        if not prompt:  # Check if user provided input
            return "What sentence or word do you want to translate?"
        else:
            try:
                # Perform translation using OpenAI API
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "system", "content": f"You are now using the {command_type} command."},
                              {"role": "user", "content": prompt}],
                    max_tokens=100
                )
                return response.choices[0].message['content'].strip()
            except Exception as e:
                return f"An error occurred: {str(e)}"
    elif command_type == "grammar":
        if not prompt:  # Check if user provided input
            return "What sentence or word do you want to check for grammar?"
        else:
            try:
                # Perform grammar check using OpenAI API
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "system", "content": f"You are now using the {command_type} command."},
                              {"role": "user", "content": prompt}],
                    max_tokens=100
                )
                return response.choices[0].message['content'].strip()
            except Exception as e:
                return f"An error occurred: {str(e)}"
    else:
        try:
            # Handle other command types
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": f"You are now using the {command_type} command."},
                          {"role": "user", "content": prompt}],
                max_tokens=100
            )
            return response.choices[0].message['content'].strip()
        except Exception as e:
            return f"An error occurred: {str(e)}"


# Accuracy of synonym and antonym predictions
def calculate_accuracy():
    # Example words
    words = ["brave", "pretty", "ugly", "shine"]
    synonym_accuracies = []
    antonym_accuracies = []

    for word in words:
        # Get synonyms and antonyms using WordNet
        true_synonyms = set(get_synonyms(word))
        true_antonyms = set(get_antonyms(word))

        # Get predicted synonyms and antonyms using models
        predicted_synonyms = set(synonym_model.predict([word])[0])
        predicted_antonyms = set(antonym_model.predict([word])[0])

        # Calculate synonym accuracy
        if true_synonyms:
            synonym_accuracy = len(true_synonyms.intersection(predicted_synonyms)) / len(true_synonyms)
            synonym_accuracies.append(synonym_accuracy)

        # Calculate antonym accuracy
        if true_antonyms:
            antonym_accuracy = len(true_antonyms.intersection(predicted_antonyms)) / len(true_antonyms)
            antonym_accuracies.append(antonym_accuracy)

    # Calculate average accuracy
    avg_synonym_accuracy = np.mean(synonym_accuracies)
    avg_antonym_accuracy = np.mean(antonym_accuracies)

    return avg_synonym_accuracy, avg_antonym_accuracy

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form['user_message']
    command_type = request.form['command_type']

    try:
        validate_command_type(command_type)
        bot_response = chat_with_bot(user_message, command_type)
        return jsonify({'bot_response': bot_response})
    except ValueError as ve:
        return jsonify({'bot_response': handle_error(str(ve))})

@app.route('/accuracy')
def accuracy():
    avg_synonym_accuracy, avg_antonym_accuracy = calculate_accuracy()
    return jsonify({'synonym_accuracy': avg_synonym_accuracy, 'antonym_accuracy': avg_antonym_accuracy})

if __name__ == '__main__':
    app.run(debug=True)
