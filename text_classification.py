from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf 
from tensorflow import keras

# Help Libraries
import numpy as np 
import matplotlib.pyplot as plt

imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

word_index = imdb.get_word_index()
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

train_data = keras.preprocessing.sequence.pad_sequences(train_data, maxlen=256, padding='post', truncating='post', value=word_index['<PAD>'])
test_data = keras.preprocessing.sequence.pad_sequences(test_data, maxlen=256, padding='post', truncating='post', value=word_index['<PAD>'])

