
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

import DataLoader.GloVe as gl
glove = gl.load('/Users/Shared/data/glove.6B/glove.6B.50d.txt')


# In[ ]:

title = df[np.arange(1,df.shape[1])]

def toVec(word):
    try:
        return glove.loc[word]
    except:
        return glove.loc['.']
    

title_vec = title.applymap(toVec)


# In[ ]:

title_expand_vec = pd.DataFrame()

for i in range(1,title_vec.shape[1]+1):
    title_expand_vec = pd.concat([title_expand_vec, title_vec[i].apply(pd.Series)], axis=1)    


# In[ ]:

result = pd.concat([title_id, title_expand_vec], axis=1)


# In[ ]:

result.to_csv('./hn_title_vec.csv', header=False, index=False)


# In[ ]:



