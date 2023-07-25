import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import tensorflow as tf
from transformers import TFBertModel
from transformers import BertTokenizer
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

with open('dff-operaomnia-lost-script.csv', mode='r', encoding='utf-8') as f:
    raw_data = f.read()

bert_model_name = "bert-base-uncased"
bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_model = TFBertModel.from_pretrained(bert_model_name)


def preprocess(data):
    tokens = bert_tokenizer.tokenize(data)
    return tokens


processed_data = [" ".join(preprocess(qa)) for qa in raw_data.split('\n')]

vocab_size = 5000
embedding_dim = 64
max_length = 128
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
training_labels = np.array(sequences)[:training_size, 1:]

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(max_length,), dtype='int32'),
    bert_model,
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
