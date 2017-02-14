
# coding: utf-8

# In[ ]:

import pandas as pd

FILE = "tokens_lower.csv"

df = pd.read_csv(FILE, header=None, dtype=str)


# In[ ]:

import numpy as np
title_id = df[0]
title_id = title_id.astype(np.int32)


# In[ ]:

title = df[np.arange(1,df.shape[1])]


# In[ ]:

import DataLoader.GloVe as gl
glove = gl.load('/Users/Shared/data/glove.6B/glove.6B.50d.txt')


# In[ ]:

from sklearn import preprocessing
import math

def toVec(word):
    if not isinstance(word, str) and np.isnan(word):
        return np.zeros(50)

    try:
        vec = glove.loc[word]
        norm_vec = preprocessing.normalize(vec.values.reshape(1,-1))
        return norm_vec[0]
    except:
        return np.zeros(50)
    
title_vec = title.applymap(toVec)


# In[ ]:

title_expand_vec = pd.DataFrame()

for i in range(1,title_vec.shape[1] + 1):
    title_expand_vec = pd.concat([title_expand_vec, title_vec[i].apply(pd.Series)], axis=1)    


# In[ ]:

result = pd.concat([title_id, title_expand_vec], axis=1)
result.columns = ['id'] + list(range(0, 1200))


# In[ ]:

result.to_csv('./hn_title_norm_vec.csv', header=True, index=False)


# In[ ]:



