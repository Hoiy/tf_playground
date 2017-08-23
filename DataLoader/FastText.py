#!/usr/bin/env python

# Return index to vector numpy array
def selective_load(file, dim, o2i, i2o, dict_size, notFound='zeros'):
    print("Start: Loading FastText Vectors")
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
