
# coding: utf-8

# In[1]:

#import Corpus.gutenberg as corpus
#from TextPreprocess.Tokenizer.Stanford import tokenize
#from Utils.visual import hist, tally
#from Utils.debug import dump
from DataLoader import GloVe
from Utils.generator import sliding_window_random_access, transform
#from Utils.FS.file import save, load
#from Utils.keras import compact_embedding
from Utils.misc import batch
#from Utils.indexer import build_index, index_2_one_hot
from Utils.indexer import build_index
#from keras.preprocessing.sequence import pad_sequences
from keras_tqdm import TQDMNotebookCallback
import numpy as np
from random import randint


# In[2]:

from nltk.corpus import gutenberg
#sents = gutenberg.sents(['austen-emma.txt', 'austen-persuasion.txt', 'austen-sense.txt'])
words = gutenberg.words(['austen-emma.txt', 'austen-persuasion.txt', 'austen-sense.txt'])
#words = gutenberg.words()
words = batch(lambda s: s.lower())(words)
print(len(words))
#sents = batch(batch(toLower))(sents)


# In[3]:

def words_generator():
    for word in words:
        yield word


# In[4]:

# build index for all words
o2i, i2o, size = build_index(words_generator())
print(size)
print(o2i('a'))
print(o2i('and'))
print(o2i('the'))


# In[5]:

#WORD_EMB_DIM = 300
#glove = GloVe.load2('./data/GloVe/glove.840B.{}d.txt'.format(WORD_EMB_DIM), WORD_EMB_DIM)

#WORD_EMB_DIM = 50
#glove = GloVe.load2('./data/GloVe/glove.6B.{}d.txt'.format(WORD_EMB_DIM), WORD_EMB_DIM)

#WORD_EMB_DIM = 50
#glove, orig_glove = GloVe.selective_load('./data/GloVe/glove.6B.{}d.txt'.format(WORD_EMB_DIM), WORD_EMB_DIM, o2i, i2o, size)

WORD_EMB_DIM = 300
glove, orig_glove = GloVe.selective_load('./data/GloVe/glove.6B.{}d.txt'.format(WORD_EMB_DIM), WORD_EMB_DIM, o2i, i2o, size)


# In[6]:

#print(np.average([len(sent) for sent in sents]))
SEQ_LENGTH = 16


# In[53]:

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

POS_EMB_SIZE = 50    

def create_encoder(inp):
    sem_emb = Embedding(glove.shape[0], glove.shape[1], weights=[glove], input_length=SEQ_LENGTH, trainable=False )(inp)
    query_emb = Embedding(glove.shape[0], POS_EMB_SIZE, embeddings_constraint=unit_norm())(inp)
    
    x = Bidirectional(GRU(POS_EMB_SIZE // 2, activation='selu', return_sequences=True))(sem_emb)
    query = Bidirectional(GRU(POS_EMB_SIZE // 2, activation='selu'))(x)
    
    return query, query_emb, sem_emb

def attention(query, query_emb, sem_emb):
    att = dot([query, query_emb], (1,2))
    att = Activation('softmax')(att)
    att = dot([att, sem_emb], (1,1))
    return att

class ReverseEmbedding(Layer):
    def __init__(self, layer, **kwargs):
        self.emb_layer = layer
        super(ReverseEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ReverseEmbedding, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return K.dot(x, K.transpose(self.emb_layer.get_weights()[0]))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.emb_layer.get_weights()[0].shape[0])


def create_baseline():
    
    GRU_DIM = 256
        
    inp = Input(shape=(SEQ_LENGTH,))
    embeddingLayer = Embedding(glove.shape[0], glove.shape[1], weights=[glove], input_length=SEQ_LENGTH, trainable=True)
    sem_emb = embeddingLayer(inp)
    x = Bidirectional(GRU(GRU_DIM, activation='selu', return_sequences=True))(sem_emb)
    predict = Bidirectional(GRU(GRU_DIM, activation='selu'))(x)
    
    #last_word = sem_emb[:,-1,:]
    
    #predict = Concatenate()([x, last_word])
    #query, query_emb, sem_emb = create_encoder(inp)
    #att = attention(query, query_emb, sem_emb)
    
    query = Dense(WORD_EMB_DIM, activation='selu')(predict)
    
    print('query', query)
    
    #predict = ReverseEmbedding(embeddingLayer)(query)
    #predict = Activation('softmax')(predict)
    
    print(predict)
    #predict = dot([])    
    #keras_glove = Input([glove.shape[0], glove.shape[1]])
    #K.is_keras_tensor(keras_input)
    
    #keras_glove = K.constant(glove, shape=(None, glove.shape[0], glove.shape[1]))
    
    #print(keras_glove)
    #print(predict)
    
    #predict = dot([predict, keras_glove], (1,1))
    
    #print(predict)
    #predict = Flatten()(predict)
    
    #print(predic)
    #predict = Activation('softmax')(predict)
    
    
    #predict = Dense(WORD_EMB_DIM, activation='selu')(predict)
    predict = Dense(glove.shape[0], activation='softmax')(query)
    model = Model(inp, predict)
    #opt = Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0002)
    model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam')
    return model


# In[54]:

model = create_baseline()
model.summary()


# In[55]:

from keras.utils.np_utils import to_categorical

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
        data.append(batch(o2i)(window[:-1]))
        label.append(o2i(window[-1]))
        if len(data) == batch_size:
            yield (np.array(data), np.array(label))
            data = []
            label = []


# In[56]:

gen = {}
size = {}
gen['train'], gen['test'], size['train'], size['test'] = sliding_window_random_access(words, SEQ_LENGTH + 1)
print(next(sample_generator(gen['train'], 2))[0].shape)
print(next(sample_generator(gen['test'], 3))[1].shape)


# In[57]:

from keras.callbacks import Callback, ModelCheckpoint

def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.random.choice(glove.shape[0], p=a)

def testing(model):
    #seed = """Your child comes home and presents you with a drawing of your house . There is a blue house , a yellow sun , and a green sky . What else do you want ?"""
    #words = seed.lower().split(' ')
    #for i in range(50):
    #    predict = model.predict(np.array([batch(o2i)(words[-SEQ_LENGTH:])]))[0]
    #    i = np.argmax(predict)
    #    words.append(i2o(i))
        
    #print(' '.join(words))
        
    seed = """Your child comes home and presents you with a drawing of your house . There is a blue house , a yellow sun , and a green sky . What else do you want ?"""
    words = seed.lower().split(' ')
    for i in range(200):
        predict = model.predict(np.array([batch(o2i)(words[-SEQ_LENGTH:])]))[0]
        i = sample(predict, 0.5)
        words.append(i2o(i))
        
    print(' '.join(words))
    
class testSample(Callback):
    def on_epoch_end(self, batch, logs={}):
        testing(model)


# In[58]:

from keras_tqdm import TQDMNotebookCallback

mc = ModelCheckpoint('./model/word_model.hdf5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

BATCH_SIZE = 2048
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

from keras.models import load_model

model = load_model('./model/word_model.hdf5')


# In[ ]:

testing(model)


# In[ ]:



