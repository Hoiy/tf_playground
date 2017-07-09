
# coding: utf-8

# # Character Level CNN

# In[1]:

import Corpus.gutenberg as corpus
from TextPreprocess.Tokenizer.Stanford import tokenize
from Utils.visual import hist, tally
from Utils.debug import dump
from Utils.generator import sliding_window, random_window, transform
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

def char_generator():
    for char in data:
        yield char


# In[4]:

s2i, i2s, size = build_index(char_generator())


# In[5]:

#MAX_SEQ_LENGTH = max([len(word) for word in data])
SEQ_LENGTH = 64


# In[6]:

NUM_SYMBOL = size
NUM_SYMBOL


# In[7]:

def word_generator():
    for word in data:
            yield word

NUM_SAMPLE = len(list(word_generator()))
NUM_SAMPLE


# In[36]:

from keras.layers import Input, Conv2D, Conv2DTranspose, Dense, Flatten, Dropout, Reshape, Embedding
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
    padding = 'same'
    use_bias = False
    layer = 3
    dim = [400, 200, 200, 200, 200, 100]
    kernel = [5, 4, 3, 2, 2, 2]
    strides = [1, 1, 1, 1, 1, ]
    
    inp = Input(shape=(SEQ_LENGTH,NUM_SYMBOL))
    x = Reshape((1, SEQ_LENGTH, NUM_SYMBOL))(inp)
    for i in range(layer):
        x = Conv2D(dim[i], (1, kernel[i]), strides=(1, strides[i]), activation=activation, padding=padding, use_bias=use_bias)(x)

    x = Dense(50, activation='selu')(x)
    x = Flatten()(x)
    x = Dense(NUM_SYMBOL, activation='softmax')(x)
    model = Model(inp, x)
    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.001)
    model.compile(loss='categorical_crossentropy',
              optimizer=opt)
    return model


# In[37]:

model = create_baseline()
model.summary()


# In[38]:

from keras.utils.np_utils import to_categorical

def encode_word(word):
    unpad = batch(s2i, word)
    one_hot = to_categorical(unpad, num_classes=NUM_SYMBOL)
    return one_hot

def decode_word(one_hot):
    return i2s( np.random.choice([i for i in range(NUM_SYMBOL)], p = one_hot)  )
    

# last char is used as output
# so set it like gen = sliding_window(SEQ_LENGTH + 1)(data)
def sample_generator(sliding_window_generator, batch_size = 64):
    data = []
    label = []
    for window in sliding_window_generator:
        data.append(encode_word(window[:-1]))
        label.append(encode_word(window[-1:])[0])
        if len(data) == batch_size:
            yield (np.array(data), np.array(label))
            data = []
            label = []


# In[39]:

window_gen = sliding_window(SEQ_LENGTH + 1)(data)
print(next(sample_generator(window_gen, 2))[0].shape)
print(next(sample_generator(window_gen, 2))[1].shape)


# In[40]:

from keras.callbacks import Callback, ModelCheckpoint
def testing(model):
    seed = """Your child comes home and presents you with a drawing of your house. There's a blue house, a yellow sun, and a green sky. You admire their handiwork and then gently ask why the sky is green. Shouldn't it be blue? Most teachers and parents would have the same reaction, but before you speak, stop! That innocent little comment carries a powerful punch. Unbeknownst to you, you are about to squelch your child's natural developing creativity.
Everyone has the ability to be creative, however, Professor of Biology and neurobiologist Erin Clabough Ph.D. writes that
"creativity can be easily crushed by goals imposed by others."
Not everyone needs to see the world in the same light- and they shouldn't. Before you mention that sky should be blue, consider your reasons carefully. Your child can see that a sky is blue, but in their world it isn't. Allow them the freedom to be creative. Creativity fosters critical thinking and problem solving skills. It helps people to deal with stress and adapt to changes.
"""
    for i in range(500):
        seed = seed + decode_word(model.predict(np.array([encode_word(seed[-SEQ_LENGTH:])]))[0])[0]
        
    print(seed)

class testSample(Callback):
    def on_epoch_end(self, batch, logs={}):
        testing(model)


# In[41]:

from keras_tqdm import TQDMNotebookCallback

mc = ModelCheckpoint('./model/char_cnn.hdf5', monitor='loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)


word_gen = word_generator()
BATCH_SIZE = 128
model.fit_generator(
    sample_generator(window_gen, BATCH_SIZE),
    NUM_SAMPLE // BATCH_SIZE // 100,
    epochs=200000,
    callbacks = [testSample(), mc]
    #verbose=0, callbacks=[TQDMNotebookCallback(), testSample()]
)


# In[ ]:

seed = """Your child comes home and presents you with a drawing of your house. There's a blue house, a yellow sun, and a green sky. You admire their handiwork and then gently ask why the sky is green. Shouldn't it be blue? Most teachers and parents would have the same reaction, but before you speak, stop! That innocent little comment carries a powerful punch. Unbeknownst to you, you are about to squelch your child's natural developing creativity.
Everyone has the ability to be creative, however, Professor of Biology and neurobiologist Erin Clabough Ph.D. writes that
"creativity can be easily crushed by goals imposed by others."
Not everyone needs to see the world in the same light- and they shouldn't. Before you mention that sky should be blue, consider your reasons carefully. Your child can see that a sky is blue, but in their world it isn't. Allow them the freedom to be creative. Creativity fosters critical thinking and problem solving skills. It helps people to deal with stress and adapt to changes.
"""

for i in range(500):
    seed = seed + decode_word(model.predict(np.array([encode_word(seed[-SEQ_LENGTH:])]))[0])[0]


# In[ ]:



