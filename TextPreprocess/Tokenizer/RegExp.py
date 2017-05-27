#!/usr/bin/env python

from nltk.tokenize import RegexpTokenizer

def tokenize(x):
    return RegexpTokenizer('\w+|\$[\d\.]+|\S+').tokenize(x)

def test():
    print(tokenize('a brown fox jumps over the lazy dog'))
    print(tokenize('1/2 Shouldn\'t "yo" hey, abc. time-out'))

if __name__=='__main__':
    test()
