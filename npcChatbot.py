import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

print(tf.__version__)

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

with open('training_data.txt', 'r', encoding='utf-8') as f:
    raw_data = f.read()


def preprocess(data):
    tokens = nltk.word_tokenize(data)

    tokens = [word.lower() for word in tokens]

    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and word not
              in string.punctuation]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return tokens


processed_data = [preprocess(qa) for qa in raw_data.split('\n')]

vocab_size = 200
embedding_dim = 100
max_length = 50
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"
training_size = len(processed_data)

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(processed_data)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(processed_data)
padded_sequences = pad_sequences(sequences, maxlen=max_length,
                                 padding=padding_type, truncating=trunc_type)

training_data = padded_sequences[:training_size]
training_labels = padded_sequences[:training_size]

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              input_length=max_length),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv1D(64, 5, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=4),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(vocab_size, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

num_epochs = 50
history = model.fit(training_data, training_labels, epochs=num_epochs,
                    verbose=2)


def predict_answer(model, tokenizer, question):
    question = preprocess(question)
    sequence = tokenizer.texts_to_sequences([question])
    padded_sequence = pad_sequences(sequence, maxlen=max_length,
                                    padding=padding_type,
                                    truncating=trunc_type)
    pred = model.predict(padded_sequence)[0]
    idx = np.argmax(pred)
    answer = tokenizer.index_word[idx + 1]
    return answer


while True:
    question = input('Player: ')
    answer = predict_answer(model, tokenizer, question)
    print('NPC:', answer)
