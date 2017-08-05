#!/usr/bin/env python

def load(file):
    from pandas import read_csv
    print("Start: Loading Glove Model")
    data = read_csv(
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

# Return index to vector numpy array
def selective_load(file, dim, o2i, i2o, dict_size, notFound='zeros'):
    print("Start: Loading Glove Model")
    f = open(file,'r', encoding='utf8')

    emb = {}
    for line in f:
        splitLine = line.split()
        vec = [float(val) for val in splitLine[-dim:]]
        emb[splitLine[0]] = vec

    print("End: Loaded %d rows." % len(emb))

    arr = []
    for i in range(dict_size):
        if i2o(i) in emb:
            arr.append(emb[i2o(i)])
        else:
            arr.append([0. for j in range(dim)])

    import numpy as np
    return np.array(arr), emb

def test():
    import os
    glove = load2(os.getcwd() + '/data/GloVe/glove.840B.300d.txt')

if __name__=='__main__':
    test()
