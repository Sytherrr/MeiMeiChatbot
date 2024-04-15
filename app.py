from flask import Flask, render_template, request, jsonify
import openai
import spacy
import numpy as np
import nltk
nltk.download('wordnet')
from keras.models import load_model
from nltk.corpus import wordnet

app = Flask(__name__)

# Set OpenAI API key (preferably through environment variable)
api_key = "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"
openai.api_key = api_key

# Load spaCy's English language model
nlp = spacy.load('en_core_web_md')

# Load the trained model
model = load_model('word_relationship_model.h5')

# Function to get word embeddings
def get_word_embedding(word):
    return nlp(word).vector

#def get_synonyms(word):
#   synonyms = []
#   for syn in wordnet.synsets(word):
#       for lemma in syn.lemmas():
#           synonyms.append(lemma.name())
#   return list(set(synonyms))

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
        return list(synonyms)

# Function to get synonyms or antonyms using the trained model
def get_synonyms_or_antonyms(word):
    global model
    word_embedding = get_word_embedding(word)
    synonyms_list = get_synonyms(word)
    if synonyms_list:
        synonyms_found = []
        for syn in synonyms_list:
            syn_embedding = get_word_embedding(syn)
            # Concatenate word embeddings
            combined_embedding = np.concatenate((word_embedding, syn_embedding))
            # Predict if the word pair is a synonym (1) or not (0)
            prediction = model.predict(np.array([combined_embedding]))[0][0]
            if prediction > 0.5:  # Threshold for considering as synonym
                synonyms_found.append(syn)
        if synonyms_found:
            return synonyms_found
        else:
            return []
    else:
        return []

    # Command type validation
def validate_command_type(command_type):
    if command_type not in ["grammar", "translate", "antonym-synonym"]:
           raise ValueError("Invalid command type")

# Error handling function
def handle_error(error):
    return f"An error occurred: {error}"

    # Function to interact with OpenAI API
def chat_with_bot(prompt, command_type):
    if command_type == "antonym-synonym":
        if not prompt:  # Check if user provided input
            return "What sentence or word do you want to inquire about?"
        else:
            try:
                word = prompt.lower()  # Convert to lowercase
                synonyms_list = get_synonyms_or_antonyms(word)
                if synonyms_list:
                    # Encode the response using UTF-8
                    synonyms_str = ', '.join(synonyms_list)
                    return f"Synonyms for '{word}': {synonyms_str}"
                else:
                    return f"No synonyms found for '{word}'."
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


if __name__ == '__main__':
    app.run(debug=True)
