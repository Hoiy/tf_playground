
# coding: utf-8

# In[ ]:

import pandas as pd
from google.cloud import storage
import os.path
import os
import gzip
import shutil

NROWS = None

client = storage.Client()
bucket = client.get_bucket('facebook-posts')

for blob in bucket.list_blobs():
    if not blob.name.lower().endswith('.gzip'):
        continue
    path = os.path.join(os.getcwd(), blob.name)
    if not os.path.exists(path):
        print('Downloading {0} to {1} ...'.format(blob.name, path))
        with open(path, 'wb') as file_obj:
            blob.download_to_file(file_obj)
        print('Downloaded {0} to {1}!'.format(blob.name, path))
    
    print('Extracting {0}'.format(path))
    with gzip.open(path, 'rb') as f_in, open(path + ".csv", 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    print('Extracted {0}'.format(path))


# In[ ]:

import pandas as pd
posts = pd.DataFrame()
for i in range(40):
    print('./posts-0000000000{:0>2d}.gzip.csv'.format(i))
    p = pd.read_csv('./posts-{:0>12d}.gzip.csv'.format(i))
    p = p[p['type'] == 'link']
    posts = posts.append(p, ignore_index=True)
    print(posts.shape)


# In[ ]:




# In[ ]:

posts.to_hdf('link_posts.h5', 'posts')


# In[5]:

import pandas as pd
posts = pd.read_hdf('./link_posts.h5', 'posts')


# In[ ]:

posts


# In[ ]:

import TextPreprocessing.Language.detector as ld
posts['lang_name'] = posts['name'].astype(str).apply(ld.detect)
#posts['lang_description'] = posts['description'].astype(str).apply(ld.detect)
#posts['lang_about'] = posts['about'].astype(str).apply(ld.detect)

#%timeit pages['name'].apply(lambda x: ld.detect(x, 'langdetect'))
#%timeit pages['about'].astype(str).apply(lambda x: ld.detect(x))
#%timeit pages['about'].astype(str).apply(lambda x: ld.detect(x, 'langdetect'))


# In[ ]:

en_link_posts = posts[posts['lang_name'] == 'en']


# In[ ]:

en_link_posts.to_hdf('link_posts.h5', 'en_link_posts')


# In[6]:

import pandas as pd
en_link_posts = pd.read_hdf('link_posts.h5', 'en_link_posts')
en_link_posts.shape


# In[7]:

posts = en_link_posts[['name', 'page_id', 'share']]


# In[8]:

import pandas as pd

NROWS = None

pages = pd.read_csv('./pages.csv', nrows=NROWS)
pages = pages[['id', 'fan_count']]


# In[9]:

result = pd.merge(posts, pages, left_on='page_id', right_on='id')


# In[10]:

result['share_ratio'] = result['share'] / result['fan_count']


# In[11]:

result['normalized_share_ratio'] = result['share_ratio'] / result['share_ratio'].max()


# In[12]:

result['normalized_share_ratio'].describe()


# In[13]:

data = result[['name', 'normalized_share_ratio']]


# In[14]:

data


# In[ ]:

from bs4 import BeautifulSoup
import re

def string_to_words( string ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(string).get_text() 
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))   


# In[ ]:

string_to_words(data['name'][0])


# In[ ]:

# Get the number of reviews based on the dataframe column size
num_name = data["name"].size

# Initialize an empty list to hold the clean reviews
clean_train_name = []

# Loop over each review; create an index i that goes from 0 to the length
# of the movie review list 
for i in range( 0, num_name ):
    # Call our function for each one, and add the result to the list of
    # clean reviews
    if( (i+1)%1000 == 0 ):
        print("Review %d of %d\n" % ( i+1, num_name ))
    clean_train_name.append( string_to_words( data["name"][i] ) )


# In[ ]:

import pickle
with open('clean_list_post_name.pkl', 'wb') as f:
    pickle.dump(clean_train_name, f)


# In[1]:

import pickle
with open('clean_list_post_name.pkl', 'rb') as f:
    clean_train_name = pickle.load(f)


# In[2]:

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


# In[ ]:

# Take a look at the words in the vocabulary
vocab = vectorizer.get_feature_names()
print(vocab)


# In[ ]:

import numpy as np

# Sum up the counts of each vocabulary word
dist = np.sum(train_data_features, axis=0)

# For each, print the vocabulary word and the number of times it 
# appears in the training set
for tag, count in zip(vocab, dist):
    print(count, tag)


# In[ ]:

import pickle
with open('clean_list_post_name_features.pkl', 'wb') as f:
    pickle.dump(train_data_features, f)


# In[15]:

print("Training the random forest...")
from sklearn.ensemble import RandomForestRegressor

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestRegressor(n_estimators = 10, max_depth=10, verbose=1, n_job=-1) 

# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
forest = forest.fit( train_data_features, data["normalized_share_ratio"] )


# In[ ]:
print('done')


