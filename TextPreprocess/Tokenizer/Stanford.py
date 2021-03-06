#!/usr/bin/env python

from nltk.tokenize import StanfordTokenizer

# https://github.com/nltk/nltk/wiki/Installing-Third-Party-Software

def tokenize(x):
    return StanfordTokenizer().tokenize(x)

def test():
    print(tokenize('a brown fox jumps over the lazy dog'))
    print(tokenize('1/2 Shouldn\'t "yo" hey, abc. time-out'))

if __name__=='__main__':
    test()
