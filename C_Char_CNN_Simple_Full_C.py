
# coding: utf-8

# # Character Level CNN

# In[1]:

import Corpus.gutenberg as corpus
from TextPreprocess.Tokenizer.Stanford import tokenize
from Utils.visual import hist, tally
from Utils.debug import dump
from Utils.generator import sliding_window_random_access, transform
from Utils.FS.file import save, load
from Utils.keras import compact_embedding
from Utils.misc import batch
from Utils.indexer import build_index, index_2_one_hot
from keras.preprocessing.sequence import pad_sequences
from keras_tqdm import TQDMNotebookCallback
import numpy as np


# In[2]:

data = corpus.raw()


# In[3]:

data = data[:len(data)]


# In[4]:

def char_generator():
    for char in data:
        yield char


# In[5]:

s2i, i2s, size = build_index(char_generator())


# In[6]:

#MAX_SEQ_LENGTH = max([len(word) for word in data])
SEQ_LENGTH = 32


# In[7]:

NUM_SYMBOL = size
NUM_SYMBOL


# In[8]:

NUM_SAMPLE = len(data)
NUM_SAMPLE


# In[9]:

from keras.layers import MaxPooling1D, Input, Conv1D, Conv2D, Conv2DTranspose, Dense, Flatten, Dropout, Reshape, Embedding, Concatenate
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras.optimizers import Adam
import numpy as np
import tensorflow as tf

def custom_loss(y_true, y_pred):
    print(y_true.shape)
    '''Just another crossentropy'''
    #y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)
    #y_pred /= y_pred.sum(axis=-1, keepdims=True)
    #cce = T.nnet.categorical_crossentropy(y_pred, y_true)
    '''
    [np.average
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y_true[i],
            logits=y_pred[i],
        )
     for i in y_true]
    '''
    return y_true - y_pred

def create_baseline():
    
    activation = 'selu'
    padding = 'valid'
    use_bias = False
    layer = 4
    dim = [100, 200, 400, 800, 1600, 3200]
    kernel = [2, 2, 2, 2, 2, 2]
    strides = [2, 2, 2, 2, 2, 2]
    EMBEDDING_SIZE=100
        
    inp = Input(shape=(SEQ_LENGTH,))
    emb = Embedding(NUM_SYMBOL, EMBEDDING_SIZE, input_length=SEQ_LENGTH)(inp)
    att_emb = Embedding(NUM_SYMBOL, EMBEDDING_SIZE, input_length=SEQ_LENGTH)(inp)
    x = emb
    for l in range(layer):
        x = Conv1D(dim[l], kernel[l], strides=strides[l], activation=activation, padding=padding, use_bias=use_bias)(x)
        #x = MaxPooling1D(strides=strides[l], padding=padding)(x)
        
    x = Flatten()(x)
    x = Dropout(0.2)(x)
    x = Dense(NUM_SYMBOL, activation='selu')(x)
    x = Dropout(0.2)(x)
    x = Dense(NUM_SYMBOL, activation='softmax')(x)
    model = Model(inp, x)
    #opt = Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0002)
    model.compile(loss='categorical_crossentropy',
              optimizer='adam')
    return model


# In[10]:

model = create_baseline()
model.summary()


# In[11]:

from keras.utils.np_utils import to_categorical

def encode_word(word):
    unpad = batch(s2i, word)
    one_hot = to_categorical(unpad, num_classes=NUM_SYMBOL)
    return one_hot

def decode_word(one_hot):
    return i2s( np.random.choice(list(range(NUM_SYMBOL)), p = one_hot)  )

def decode_word_2(one_hot):
    return i2s( np.argmax(one_hot) )


# last char is used as output
# so set it like gen = sliding_window(SEQ_LENGTH + 1)(data)
def sample_generator(sliding_window_generator, batch_size = 64):
    data = []
    label = []
    for window in sliding_window_generator:
        #data.append(encode_word(window[:-1]))
        data.append(batch(s2i, window[:-1]))
        label.append(encode_word(window[-1:])[0])
        if len(data) == batch_size:
            yield (np.array(data), np.array(label))
            data = []
            label = []


# In[12]:

gen = {}
size = {}
gen['train'], gen['test'], size['train'], size['test'] = sliding_window_random_access(data, SEQ_LENGTH + 1)
print(next(sample_generator(gen['train'], 2))[0].shape)
print(next(sample_generator(gen['test'], 2))[1].shape)


# In[13]:

from keras.callbacks import Callback, ModelCheckpoint
def testing(model):
    seed = """Your child comes home and presents you with a drawing of your house. There's a blue house, a yellow sun, and a green sky. You admire their handiwork and then gently ask why the sky is green. Shouldn't it be blue? Most teachers and parents would have the same reaction, but before you speak, stop! That innocent little comment carries a powerful punch. Unbeknownst to you, you are about to squelch your child's natural developing creativity.
Everyone has the ability to be creative, however, Professor of Biology and neurobiologist Erin Clabough Ph.D. writes that "creativity can be easily crushed by goals imposed by others." Not everyone needs to see the world in the same light- and they shouldn't. Before you mention that sky should be blue, consider your reasons carefully. Your child can see that a sky is blue, but in their world it isn't. Allow them the freedom to be creative. Creativity fosters critical thinking and problem solving skills. It helps people to deal with stress and adapt to changes.
"""
    for i in range(500):
        seed = seed + decode_word(model.predict(np.array([batch(s2i,seed[-SEQ_LENGTH:])]))[0])[0]
        
    print(seed)

    seed = """Your child comes home and presents you with a drawing of your house. There's a blue house, a yellow sun, and a green sky. You admire their handiwork and then gently ask why the sky is green. Shouldn't it be blue? Most teachers and parents would have the same reaction, but before you speak, stop! That innocent little comment carries a powerful punch. Unbeknownst to you, you are about to squelch your child's natural developing creativity.
Everyone has the ability to be creative, however, Professor of Biology and neurobiologist Erin Clabough Ph.D. writes that "creativity can be easily crushed by goals imposed by others." Not everyone needs to see the world in the same light- and they shouldn't. Before you mention that sky should be blue, consider your reasons carefully. Your child can see that a sky is blue, but in their world it isn't. Allow them the freedom to be creative. Creativity fosters critical thinking and problem solving skills. It helps people to deal with stress and adapt to changes.
"""
    for i in range(500):
        seed = seed + decode_word_2(model.predict(np.array([batch(s2i,seed[-SEQ_LENGTH:])]))[0])[0]
        
    print(seed)

    
class testSample(Callback):
    def on_epoch_end(self, batch, logs={}):
        testing(model)


# In[ ]:

from keras_tqdm import TQDMNotebookCallback

mc = ModelCheckpoint('./model/char_cnn_2.hdf5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

BATCH_SIZE = 1024
model.fit_generator(
    sample_generator(gen['train'], BATCH_SIZE),
    size['train'] // BATCH_SIZE,
    validation_data = sample_generator(gen['test'], BATCH_SIZE),
    validation_steps = size['test'] // BATCH_SIZE,
    epochs=200000,
    callbacks = [testSample(), mc]
    #verbose=0, callbacks=[TQDMNotebookCallback(), testSample()]
)


# In[ ]:




# In[ ]:



