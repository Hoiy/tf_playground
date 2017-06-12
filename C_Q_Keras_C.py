
# coding: utf-8

# In[1]:

from Utils.FS import file
from Utils.tensorflow_helper import show_graph
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import brown
from scipy.sparse import coo_matrix, dok_matrix
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from nltk.corpus import stopwords
import tensorflow as tf
import math
import TextPreprocess.words2dict as words2dict
from tensorflow.python.layers import core as layers_core
from tensorflow.python.client import timeline
import time
from DataLoader import GloVe
from TextPreprocess.sequences import Sequences
from TextPreprocess.Tokenizer.RegExp import tokenize
import Utils.pandas_helper as ph


# In[2]:

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# In[3]:

np.random.seed(1234)
WORD_DIM = 300
WORD_COUNT = 400000+3


# In[4]:

glove = GloVe.load2('./data/GloVe/glove.6B.{}d.txt'.format(WORD_DIM))


# In[5]:

# emb: Symbol to float32 of fixed DIMENSION
# Create an index mapping, index to symbol, symbol to index

class Embedding:
    def __init__(self, emb, verbose = False):
        # assert emb is dictionary and each entry has same dimension
        self.emb = emb
        self.dim = len(self.emb[list(self.emb.keys())[0]])
        self.emb['<UNK>'] = [0. for i in range(self.dim)]
        self.emb['<PAD>'] = [1. for i in range(self.dim)]
        self.emb['<GO>'] = [-1. for i in range(self.dim)]
        
        self.build_dicts()
        
        if verbose:
            self.describe()
        
    def describe(self):
        print('Embedding Dimension: {}'.format(self.dim))
        print('Embedding Symbols: {}'.format(len(self.emb)))
        print('Index to symbol: {}'.format([(i, self.idx2Sym[i]) for i in range(10)]))
        
    def getIndex(self, symbol):
        if symbol in self.sym2Idx:
            return self.sym2Idx[symbol]
        else:
            return self.sym2Idx['<UNK>']

    def getEmb(self, symbol):
        return self.emb[self.idx2Sym[self.getIndex(symbol)]]
    
    def getSymbols(self, indices):
        return [self.idx2Sym[idx] for idx in indices]

    def getNumpyArray(self):
        return np.array([self.emb[self.idx2Sym[idx]] for idx in range(len(self.emb))])
    
    def build_dicts(self):
        self.sym2Idx = {}
        index = 0
        for key in sorted(self.emb.keys()):
            self.sym2Idx[key] = index
            index += 1
            
        self.idx2Sym = { v:k for k, v in self.sym2Idx.items()}

glove_emb = Embedding(glove, verbose=True)


# In[6]:

df = file.read('data/Quora/train.csv')

from sklearn.model_selection import train_test_split

df.question1 = df.question1.astype(str)
df.question2 = df.question2.astype(str)
df.is_duplicate = df.is_duplicate.astype(float)

df = df.as_matrix(['question1', 'question2', 'is_duplicate'])

data = {}
data['train'], data['test'] = train_test_split(df, test_size = 0.1)


# In[7]:

def preprocessQuestion(string):
    try:
        return [glove_emb.getIndex(token.lower()) for token in tokenize(string)]
    except:
        print(string)


def preprocessData(data):
    return [[preprocessQuestion(rec[0]), preprocessQuestion(rec[1]), float(rec[2])] for rec in data]


# In[8]:

for i in ['train', 'test']:
    data[i] = preprocessData(data[i])


# In[9]:

# Turns iteratable of symbols into padded batch
from functools import lru_cache

class Batcher:
    def __init__(self, sequences, verbose = False):
        self.seqs = sequences
        self.verbose = verbose
        self.size = len(self.seqs)
        self.seq_lens = [len(seq) for seq in self.seqs]
        
        if self.verbose:
            self.describe()
    
    @lru_cache(maxsize=None)
    def max_length(self):
        return max(self.seq_lens)
    
    @lru_cache(maxsize=None)
    def longgest_sequence(self):
        for seq in self.seqs:
            if len(seq) == self.max_length():
                return seq
    
    def describe(self):
        print('Size: {}'.format(self.size))
        print("Longest sequence length: {}".format(self.max_length()))
        bin_width = max(1, self.max_length() // 30)
        plt.hist(self.seq_lens, range(0, self.max_length() + bin_width, bin_width))
        plt.title('Sequence length distribution')
        plt.show()
        
    def batchPadding(self, batch, padding_symbol):
        size = max([len(record) for record in batch])
        result = np.full((len(batch), size), padding_symbol)
        for i in range(len(batch)):
            result[i][:len(batch[i])] = batch[i]
        return result

    def batchMask(self, batch):
        size = max([len(record) for record in batch])
        result = np.full((len(batch), size), 0.0)
        for i in range(len(batch)):
            result[i][:len(batch[i])] = 1.0
        return result
        
    # Same length within the batch, stuffed with padding symbol
    def generator(self, padding_symbol, batch_size=None, epouch=-1):
        if batch_size == None:
            batch_size = self.size
        train = []
        length = []
        while(epouch < 0 or epouch > 0):
            for seq in self.seqs:
                train.append([sym for sym in seq])
                length.append(len(seq))
                if(len(train) == batch_size):
                    yield self.batchPadding(train, padding_symbol), length, self.batchMask(train)
                    train = []
                    length = []
            epouch -= 1
            if self.verbose:
                print('epouch done...')
                
class Batcher2:
    def __init__(self, sequences, verbose = False):
        self.seqs = sequences
        self.size = len(self.seqs)

    def generator(self, batch_size=32, epouch=-1):
        if batch_size == None:
            batch_size = self.size
        train = []
        while(epouch < 0 or epouch > 0):
            for sym in self.seqs:
                train.append([sym])
                if(len(train) == batch_size):
                    yield train
                    train = []
            epouch -= 1
            print('epouch done...')
            
            
# Turn data into batch, where data is iterable over records, record is iterable over fields
class Batcher3:
    def __init__(self, data):
        #assert it is doubly iterable
        self.data = data
        self.size = len(data)
        
    def generator(self, batch_size = 32, epouch = -1):
        batch = []
        while(epouch < 0 or epouch > 0):
            for record in self.data:
                batch.append(record)
                if(len(batch) == batch_size):
                    yield batch
                    batch = []
            epouch -= 1
            print('epouch done...')


# In[10]:

batcher = {}
for i in ['train', 'test']:
    batcher[i] = Batcher3(data[i])


# In[11]:

LV1_DIM = 10
LV2_STEP = 1
LV2_DIM = 150

DROPOUT_RNN = 0.1
DROP_DENSE = 0.1


# In[12]:

def embeddings_initializer(shape):
    with tf.variable_scope("Embeddings_Initializer"):
        in_emb = tf.placeholder(
            dtype = tf.float32, 
            shape = shape, 
            name = "Placeholder"
        )
        
        emb = tf.Variable(
            tf.constant(0.0, shape = shape), 
            trainable=False, 
            name = 'Embeddings', 
            dtype=tf.float32
        )
        
        init_emb = emb.assign(in_emb)
    return in_emb, init_emb, emb

def cells_initializer(num_units, reuse):
    with tf.variable_scope("Cells_Initializer"):
        cells = tf.contrib.rnn.GRUCell(
            num_units = num_units,
            input_size = None,
            activation = tf.tanh,
            reuse = reuse
        )
    return cells


#IN (batch, time, 1)
def simple_embedding(inputs, embeddings):
    with tf.variable_scope("Simple_Embedding"):
        lookup = tf.nn.embedding_lookup(
            params = embeddings,
            ids = inputs,
            partition_strategy='mod',
            name='Embedding_Lookup',
            validate_indices=True,
            max_norm=None
        )

    return lookup

#OUT: (batch, time, dim) float32

#IN (batch, time, dim)
def simple_dynamic_rnn(cell, inputs, lengths):
    with tf.variable_scope("Simple_Dynamic_RNN"):        
        
        batch_size = tf.shape(inputs)[0]
        step_size = tf.shape(inputs)[1]

        outputs, states = tf.nn.dynamic_rnn(
            cell, 
            inputs, 
            dtype = tf.float32, 
            sequence_length = lengths,
            initial_state = cell.zero_state(batch_size, dtype=tf.float32)
        )

        indices = tf.range(0, batch_size) * step_size + (lengths - 1)
        gather = tf.reshape(
            tf.gather(
                tf.reshape(outputs, [-1, cell.output_size]), indices
            ), 
            [-1, cell.output_size]
         )
        
    return gather
#OUT (batch, dim)

#IN (batch, time, dim)
def simple_encoder(inputs, input_lengths, embeddings, dropout = 0.0, reuse = None):
    with tf.variable_scope('Simple_Encoder'):
        
        emb = simple_embedding(inputs, embeddings)
        
        cell = tf.contrib.rnn.GRUCell(
            num_units = LV2_DIM,
            input_size = None,
            activation = tf.tanh,
            reuse = reuse
        )
        
        cell = tf.contrib.rnn.DropoutWrapper(
            cell,
            input_keep_prob = 1. - dropout,
            output_keep_prob = 1. - dropout,
            state_keep_prob = 1. - dropout,
            variational_recurrent=False,
            input_size=None,
            dtype=None,
            seed=None
        )

        rnn = simple_dynamic_rnn(
            cell = cell,
            inputs = emb,
            lengths = input_lengths
        )
        
    return rnn, emb
            
        #
        # Conv layer does not support dynamic length ;/
        #
    """
        filter_2 = tf.Variable(
            tf.random_uniform([2, WORD_DIM, LV1_DIM], -1, 1), 
            dtype=tf.float32
        )

        #IN (batch, time, dim)
        conv_2 = tf.nn.conv1d(
            value = inputs,
            filters = filter_2,
            stride = 1,
            padding = 'VALID',
            use_cudnn_on_gpu=True,
            data_format=None,
            name='Conv_Witdh_2'
        )
        #OUT (batch, time-1, dim)

    with tf.variable_scope('Level_2_RNN'):
        
        cell = tf.contrib.rnn.GRUCell(
            num_units = LV2_DIM,
            input_size=None,
            activation=tf.tanh,
            reuse = reuse
        )
        
        rnn_output_2 = simple_dynamic_rnn(
            cell = cell,
            inputs = inputs,
            lengths = input_lengths
        )
        
    return rnn_output_2

    """
    
#OUT (batch, dim)


# In[13]:

import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Concatenate, Reshape, concatenate
from keras.layers import Conv1D, MaxPooling1D, Bidirectional, GRU
from keras.models import Model
from keras.layers import Flatten
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
MAX_SEQUENCE_LENGTH = 30
num_dense = 50


# In[27]:

conv_max_length = 2
conv_dim = 100
rnn_step = 5
conv_layers = 2

print(conv_max_length)
print(conv_dim)
print(rnn_step)

########################################
## define the model structure
########################################
embedding_layer = Embedding(WORD_COUNT,
        WORD_DIM,
        weights=[glove_emb.getNumpyArray()],
        input_length=MAX_SEQUENCE_LENGTH,
        trainable=False)

lstm_layer = Bidirectional(GRU(200, dropout=0.2, recurrent_dropout=0.2))
#lstm_layer = Bidirectional(GRU(200, dropout=0.1, recurrent_dropout=0.1))(lstm_layer)
lstm_layer_0 = [Bidirectional(GRU(100, dropout=0.2, recurrent_dropout=0.2)) for i in range(rnn_step)]

sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences_1 = embedding_layer(sequence_1_input)

# create same conv for both
conv = [[Conv1D((j+1) * 2 * WORD_DIM, 2, padding='same') for j in range(conv_layers)] for i in range(rnn_step)]
maxpool = [[MaxPooling1D(pool_size=2, strides=2, padding='same') for j in range(conv_layers)] for i in range(rnn_step)]
flatten = Flatten()
dense = [Dense(600, activation='relu') for j in range(conv_layers)]
reshape = Reshape((1, -1))

def pipe(embedded_sequences):
    steps = [lstm_layer_0[i](embedded_sequences) for i in range(rnn_step)]
    reshapes = [Reshape((1, -1))(steps[i]) for i in range(rnn_step)]
    print(reshapes)
    concat = Concatenate(1)(reshapes)
    print(concat)
    feat = lstm_layer(concat)
    #feat = conv[0][0](embedded_sequences)
    #feat = maxpool[0][0](feat)
    #feat = conv[0][1](feat)
    #feat = maxpool[0][1](feat)
    
    #feat = [Reshape((1, -1))(Concatenate()(feat[i])) for i in range(rnn_step)]
    #feat = Concatenate(1)(feat)
    #lstm_layer(feat)
    print(feat)
    #feat = flatten(feat)
    #feat = dense(feat)
    return feat

x1 = pipe(embedded_sequences_1)

sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences_2 = embedding_layer(sequence_2_input)

y1 = pipe(embedded_sequences_2)

merged = Concatenate()([x1, y1])
merged = Dropout(0.2)(merged)
merged = BatchNormalization()(merged)

merged = Dense(num_dense, activation='relu')(merged)
merged = Dropout(0.2)(merged)
merged = BatchNormalization()(merged)

preds = Dense(1, activation='sigmoid')(merged)


# In[15]:

""" 0.49 LB

embedding_layer = Embedding(WORD_COUNT,
        WORD_DIM,
        weights=[glove_emb.getNumpyArray()],
        input_length=MAX_SEQUENCE_LENGTH,
        trainable=False)

lstm_layer = LSTM(150, dropout=0.3, recurrent_dropout=0.3)

sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences_1 = embedding_layer(sequence_1_input)
x1 = lstm_layer(embedded_sequences_1)

sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences_2 = embedding_layer(sequence_2_input)
y1 = lstm_layer(embedded_sequences_2)

merged = concatenate([x1, y1])
merged = BatchNormalization()(merged)
merged = Dropout(0.3)(merged)

merged = Dense(num_dense, activation='relu')(merged)
merged = BatchNormalization()(merged)
merged = Dropout(0.3)(merged)

preds = Dense(1, activation='sigmoid')(merged)
"""


# In[16]:

model = Model(
    inputs=[sequence_1_input, sequence_2_input],
    outputs=preds
)
model.compile(loss='binary_crossentropy',
        optimizer='adam',
        metrics=['acc'])


# In[17]:

q1 = {}
q2 = {}
label = {}

for i in ['train', 'test']:
    q1[i] = [rec[0] for rec in data[i]]
    q1[i] += [rec[1] for rec in data[i]]
    
    q1[i] = pad_sequences(q1[i], maxlen=MAX_SEQUENCE_LENGTH)
    
    q2[i] = [rec[1] for rec in data[i]]
    q2[i] += [rec[0] for rec in data[i]]
    
    q2[i] = pad_sequences(q2[i], maxlen=MAX_SEQUENCE_LENGTH)
    
    label[i] = [rec[2] for rec in data[i]]
    label[i] += [rec[2] for rec in data[i]]


# In[18]:

print(len(q1['train']))
print(len(q2['train']))
print(len(label['train']))

weight_val = np.ones(len(label['test']))
weight_val *= 0.472001959
for i in range(len(label['test'])):
    if label['test'][i] == 0:
        weight_val[i] = 1.309028344

class_weight = {0: 1.309028344, 1: 0.472001959}

print(label['test'][:10])
print(weight_val[:10])


# In[19]:

positive_ratio = sum(label['train']) / len(label['train'])


# In[ ]:

#import keras
early_stopping =EarlyStopping(monitor='val_loss', patience=3)
bst_model_path = 'q.h5'
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)
tbCallBack = keras.callbacks.TensorBoard(log_dir='./Tensorboard', histogram_freq=0, write_graph=True, write_images=True)

hist = model.fit(
    [q1['train'], q2['train']], 
    label['train'],
    validation_data=([q1['test'], q2['test']], label['test'], weight_val),
    epochs=200, 
    batch_size=512, 
    shuffle=True,
    #class_weight = {0: 0.5 / (1. - positive_ratio), 1: 0.5 / positive_ratio },
    class_weight = class_weight,
    callbacks=[tbCallBack, early_stopping, model_checkpoint]
)


# In[23]:

test_data = file.read('./data/Quora/test.csv')
test_data.question1 = test_data.question1.astype(str)
test_data.question2 = test_data.question2.astype(str)
test_data = test_data.as_matrix(['test_id', 'question1', 'question2'])


# In[24]:

test_ids = [rec[0] for rec in test_data]


# In[25]:

test_data_1 = []
test_data_2 = []
for rec in test_data:
    test_data_1.append(preprocessQuestion(rec[1]))
    test_data_2.append(preprocessQuestion(rec[2]))
    
test_data_1 = pad_sequences(test_data_1, maxlen=MAX_SEQUENCE_LENGTH)
test_data_2 = pad_sequences(test_data_2, maxlen=MAX_SEQUENCE_LENGTH)


# In[26]:

bst_model_path = 'q.h5'
model.load_weights(bst_model_path)
#bst_val_score = min(hist.history['val_loss'])

########################################
## make the submission
########################################
print('Start making the submission before fine-tuning')

preds = model.predict([test_data_1, test_data_2], batch_size=8192, verbose=1)
#preds += model.predict([test_data_2, test_data_1], batch_size=8192, verbose=1)
#preds /= 2


# In[ ]:

import pandas as pd
submission = pd.DataFrame({'test_id':test_ids, 'is_duplicate': preds.ravel()})
submission.to_csv('q.csv', index=False)


# In[ ]:



