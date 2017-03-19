
# coding: utf-8

# In[1]:

import pickle
import pandas as pd


# In[2]:

with open('eng_link_posts.pkl', 'rb') as f:
    en_link_posts = pickle.load(f)


# In[3]:

NROWS = None

pages = pd.read_csv('./pages.csv', nrows=NROWS)
pages = pages[['id', 'fan_count']]


# In[4]:

result = pd.merge(en_link_posts, pages, left_on='page_id', right_on='id')


# In[5]:

result['share_ratio'] = result['share'] / result['fan_count']
result['normalized_share_ratio'] = result['share_ratio'] / result['share_ratio'].max()
print(result['normalized_share_ratio'].describe())
data = result[['name', 'normalized_share_ratio']]


# In[6]:

import pickle
with open('clean_list_post_name.pkl', 'rb') as f:
    clean_train_name = pickle.load(f)


# In[7]:

print("Creating the bag of words...\n")
from sklearn.feature_extraction.text import CountVectorizer

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer = CountVectorizer(analyzer = "word",                                tokenizer = None,                                 preprocessor = None,                              stop_words = None,                                max_features = 5000) 

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
train_data_features = vectorizer.fit_transform(clean_train_name)

# Numpy arrays are easy to work with, so convert the result to an 
# array
train_data_features = train_data_features.toarray()


# In[8]:

vocab = vectorizer.get_feature_names()


# In[9]:

import numpy as np
train_X = train_data_features[:300000, :]
train_y = data["normalized_share_ratio"].values[:300000]


# import xgboost as xgb
# 
# 
# xgb_params = {"objective": "reg:linear", "eta": 0.01, "max_depth": 10, "seed": 42, "silent": 1, "booster":"gblinear"}
# num_rounds = 100
# 
# dtrain = xgb.DMatrix(train_X, label=train_y, feature_names=vocab)
# gbm = xgb.train(xgb_params, dtrain, num_rounds)
# 
# dtest = xgb.DMatrix(train_X, feature_names=vocab)
# 
# print(gbm.predict(dtest))
# 
# print(gbm.get_score())
# print(gbm.get_fscore())

# In[13]:

import xgboost as xgb
gbm = xgb.XGBRegressor().fit(train_X, train_y)


# In[12]:

importances = list(zip(vocab, gbm.feature_importances_))
importances.sort(key = lambda t: t[1], reverse=True)
print(importances[:100])


# In[ ]:



