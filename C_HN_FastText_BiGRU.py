
# coding: utf-8

# In[1]:

import pandas as pd
from Utils.misc import batch


# In[2]:

data = pd.read_csv("./data/HN/HN_posts_year_to_Sep_26_2016.csv", parse_dates=['created_at'])
data = data[["title", "num_points"]]


# In[3]:

data['num_points'].describe()


# In[4]:

len(data)


# In[5]:

data_mat = data.as_matrix()


# In[6]:

def wordGen():
    for i in range(len(data)):
        for word in batch(lambda x: x.lower())(data.iloc[i]['title'].split(' ')):
            yield word
    return


# In[7]:

count = 0
for i in range(len(data)):
    count = count + (0 if data_mat[i][1] <= 4 else 1)
print(1- (count / len(data)) )


# In[ ]:

from Utils.indexer import build_index

o2i, i2o, size = build_index(wordGen())
print(size)


# In[ ]:

from DataLoader import FastText

WORD_EMB_DIM = 300
ft, orig_ft = FastText.selective_load('./data/FastText/wiki.en.vec', WORD_EMB_DIM, o2i, i2o, size)


# In[ ]:

SEQ_LENGTH = 30


# In[ ]:

from keras.layers import Activation, dot, add, MaxPooling1D, MaxPooling2D, Bidirectional, Input, GRU, LSTM, SimpleRNN, Conv1D, Conv2D, Conv2DTranspose, Dense, Flatten, Dropout, Reshape, Embedding, Concatenate
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.constraints import unit_norm
from keras.initializers import Identity
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf

def create_baseline():
    
    GRU_DIM = 512
        
    inp = Input(shape=(SEQ_LENGTH,))
    sem_emb = Embedding(ft.shape[0], ft.shape[1], weights=[ft], input_length=SEQ_LENGTH, trainable=False)(inp)
    
    x = Bidirectional(GRU(GRU_DIM // 2, activation='selu', return_sequences=True))(sem_emb)
    x = Bidirectional(GRU(GRU_DIM // 2, activation='selu', return_sequences=True))(sem_emb)
    x = Bidirectional(GRU(GRU_DIM // 2, activation='selu'))(x)

    predict = Dense(1, activation='sigmoid')(x)
    model = Model(inp, predict)
    model.compile(loss='binary_crossentropy',
        optimizer='adam',
        metrics=['acc'])
    return model


# In[ ]:

model = create_baseline()
model.summary()


# In[ ]:

from Utils.generator import random_access

gen = {}
size = {}
gen['train'], gen['test'], size['train'], size['test'] = random_access(data_mat)


# In[ ]:

from keras.preprocessing.sequence import pad_sequences

def sample_generator(gen, batch_size = 64):
    data = []
    label = []
    for row in gen:
        data.append(batch(o2i)(batch(lambda x: x.lower())(row[0].split(' '))))
        lab = 0 if row[1] <= 4 else 1
        label.append([lab])
        if len(data) == batch_size:
            yield (pad_sequences(np.array(data), maxlen=SEQ_LENGTH), np.array(label))
            data = []
            label = []


# In[ ]:

print(next(sample_generator(gen['train'], 2))[0].shape)
print(next(sample_generator(gen['test'], 3))[1].shape)


# In[ ]:

from keras_tqdm import TQDMNotebookCallback
from keras.callbacks import Callback, ModelCheckpoint

mc = ModelCheckpoint('./model/hn_fasttext_model.hdf5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

BATCH_SIZE = 1024
model.fit_generator(
    sample_generator(gen['train'], BATCH_SIZE),
    size['train'] // BATCH_SIZE,
    validation_data = sample_generator(gen['test'], BATCH_SIZE),
    validation_steps = size['test'] // BATCH_SIZE,
    epochs=200000,
    callbacks = [mc]
    #verbose=0, callbacks=[TQDMNotebookCallback(),mc]
)


# In[ ]:




# In[ ]:



