# train_model.py
import json
import numpy as np
import spacy
from keras.models import Sequential
from keras.layers import Dense

# Load the JSON file containing word mappings
with open('dataset/words_map.json', 'r') as file:
    word_map = json.load(file)

# Process the word mappings to extract synonyms and antonyms
synonyms = {}
antonyms = {}

for word, data in word_map.items():
    if 'synonyms' in data:
        synonyms[word] = data['synonyms']
    if 'antonyms' in data:
        antonyms[word] = data['antonyms']

# Load spaCy's English language model
nlp = spacy.load('en_core_web_md')

# Function to get word embeddings
def get_word_embedding(word):
    if isinstance(word, float):
        # Handle float values appropriately, for example, convert them to strings
        word = str(word)
    return nlp(word).vector

# Prepare training data
X_train = []
y_train = []

for word, data in word_map.items():
    if isinstance(data, dict):  # Check if data is a dictionary
        syns = data.get('synonyms', [])  # Get synonyms, or an empty list if not present
        ants = data.get('antonyms', [])  # Get antonyms, or an empty list if not present

        # Ensure syns and ants are lists
        if not isinstance(syns, list):
            syns = [syns]  # Convert to list if it's not already one
        if not isinstance(ants, list):
            ants = [ants]  # Convert to list if it's not already one

        for syn in syns:
            # Get word embeddings
            word_embedding = get_word_embedding(word)
            syn_embedding = get_word_embedding(syn)
            # Concatenate embeddings
            concatenated_embedding = np.concatenate((word_embedding, syn_embedding))
            X_train.append(concatenated_embedding)
            y_train.append(1)

        for ant in ants:
            # Get word embeddings
            word_embedding = get_word_embedding(word)
            ant_embedding = get_word_embedding(ant)
            # Concatenate embeddings
            concatenated_embedding = np.concatenate((word_embedding, ant_embedding))
            X_train.append(concatenated_embedding)
            y_train.append(0)
    else:
        print(f"Warning: Unexpected data type for '{word}' - Skipping...")

# Convert lists to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

# Ensure the concatenated embeddings have the correct shape
assert X_train.shape[1] == 600, "Concatenated embeddings have incorrect shape"

# Define the neural network model
model = Sequential([
    Dense(128, input_dim=600, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Save the trained model
model.save('word_relationship_model.h5')
