
# coding: utf-8

# In[9]:

import pandas as pd

FILE = "/Users/Shared/data/HN_posts_year_to_Sep_26_2016.csv"

data = pd.read_csv(FILE)
data = data[["id", "title"]]


# In[12]:

from nltk.tokenize import StanfordTokenizer
import numpy as np

def tokenize(x):
    print(x)
    return pd.Series(StanfordTokenizer().tokenize(x))

title = data["title"].apply(lambda x: tokenize(x))


# In[14]:

result = pd.concat([data["id"], title], axis=1)


# In[15]:

result.to_csv('./tokens.csv', header=False, index=False)

