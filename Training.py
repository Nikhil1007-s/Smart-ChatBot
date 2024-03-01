import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

# Initialize WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents from JSON file
with open("intents.json", 'r') as file:
    intents = json.load(file)

# Initialize lists for words, classes, and documents
words = []
classes = []
documents = []
ignore_letters = ['!', '?', '.', ',']

# Tokenization and lemmatization of intents
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize words and remove duplicates
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))

# Save words and classes
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Initialize training data
training = []
output_empty = [0] * len(classes)

# Create bag of words for each document
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

# Define model architecture
# Define model architecture
model = Sequential()
model.add(Dense(128, input_shape=(len(training[0][0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(training[0][1]), activation='softmax'))


# Compile model
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train model
batch_size = 5
epochs = 200
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    np.random.shuffle(training)
    for i in range(0, len(training), batch_size):
        batch_x = np.array(training[i:i+batch_size, 0].tolist())  # Extract features
        batch_y = np.array(training[i:i+batch_size, 1].tolist())  # Extract labels
        model.train_on_batch(batch_x, batch_y)

# Save model
model.save('chatbotmodel.h5')
print("Model saved.")

