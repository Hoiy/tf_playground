#!/usr/bin/env python

from spacy.en import English
nlp = English()

def tokenize(x):
    return [token.text for token in nlp(x)]

def test():
    print(tokenize('a brown fox jumps over the lazy dog'))
    print(tokenize('1/2 Shouldn\'t "yo" hey, abc. time-out'))

if __name__=='__main__':
    test()
