#!/usr/bin/env python

import pandas as pd

def load(file):
    print("Start: Loading Glove Model")
    data = pd.read_csv(
        file,
        header=None,
        index_col=0,
        delim_whitespace=True,
        quoting=3
    )
    print("End: Loaded %d rows." % data.shape[0])
    return data

def load2(file = '/Users/Shared/data/glove.6B/glove.6B.50d.txt', dim=50):
    print("Start: Loading Glove Model")
    f = open(file,'r', encoding='utf8')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        try:
            embedding = [float(val) for val in splitLine[1:dim+1]]
            model[word] = embedding
        except:
            pass

    print("End: Loaded %d rows." % len(model))
    return model

def test():
    import os
    glove = load2(os.getcwd() + '/data/GloVe/glove.840B.300d.txt')

if __name__=='__main__':
    test()
