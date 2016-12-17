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

def test():
    glove = load('/Users/Shared/data/glove.6B/glove.6B.50d.txt')
    print(glove)
    print(glove.loc['the'])

if __name__=='__main__':
    test()
