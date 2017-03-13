#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

EMBEDDING_DIM = 300
MAX_SEQUENCE_LENGTH = 30

def load_wordvector():
    embeddings_index = {}
    f = open(os.path.join('/Users/Shared/data/glove.6B/', 'glove.6B.' + str(EMBEDDING_DIM) + 'd.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index

def build_embedding_matrix(word_index, embeddings_index):
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    print('Built embedding_matrix')
    return embedding_matrix

def tokenize(texts):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    print('Found %s unique tokens.' % len(word_index))
    return word_index, padded_sequences

def expand_sequences(pad_sequences, embedding_matrix):
    expand_sequence = np.zeros((pad_sequences.shape[0], pad_sequences.shape[1] * EMBEDDING_DIM))
    for i in range(0, pad_sequences.shape[0]):
        for j in range(0, pad_sequences.shape[1]):
            expand_sequence[i][j*EMBEDDING_DIM:(j+1)*EMBEDDING_DIM] = embedding_matrix[x_train[i][j]]
    return expand_sequence

# Assumption:
# pdData must contains pdData['data'] and pdData['label']
# pdData['data'] is string
# pdData['label'] is binary 0 or 1
def test_data(pdData):
    word_index, padded_sequences = tokenize(pdData['data'])
    embeddings_index = load_wordvector()
    embedding_matrix = build_embedding_matrix(word_index, embeddings_index)
    pdData["data"] = expand_sequences(padded_sequences, embedding_matrix)
    #x_train = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    train = pdData.sample(frac=0.8)
    test = pdData.drop(train.index)

    rf = RandomForestClassifier(n_estimators=2000, n_jobs=-1, criterion="entropy", random_state=1)
    rf.fit(train["data"], train["label"])
    res = rf.predict(text["data"])

    precision = precision_score(test["label"], res)
    recall = recall_score(test["label"], res)
    print(precision)
    print(recall)




def test():
    print('hi')

if __name__=='__main__':
    test()
