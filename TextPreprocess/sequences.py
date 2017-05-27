from functools import lru_cache
from collections import Counter
import numpy as np
# Sequences is an iterable of sequence, sequence is an iterable of symbol
# This class build a lookup table for mapping symbol to an index and vice versa
# By default, this lookup table sort the frequency of symbol in descending order.
# This is required for using, for example, nce_loss function of tensorflow use uni-log sampling.

class Sequences:
    def __init__(self, sequences=[], verbose=False):
        # assert sequences is iterable and each sequence is also iterable
        self.seqs = sequences
        self.verbose = verbose
        self.build_dicts()
        if self.verbose:
            self.describe()
                
    @lru_cache(maxsize=None)
    def max_length(self):
        return max([len(seq) for seq in self.seqs])
    
    # build forward and backward lookup dictionaries for symbols to indices
    def build_dicts(self):
        if self.verbose:
            print("Building dictionaries...")
        
        counter = Counter([symbol for seq in self.seqs for symbol in seq])
        self.sym2Idx = {}
        index = 0
        for item in sorted(counter.items(), key=lambda x: (x[1], x[0]), reverse=True):
            self.sym2Idx[item[0]] = index
            index += 1

        self.UNK = '<UNK>'
        self.sym2Idx[self.UNK] = index
        index += 1

        self.PAD = '<PAD>'
        self.sym2Idx[self.PAD] = index
        index += 1

        self.GO = '<GO>'
        self.sym2Idx[self.GO] = index
        index += 1
            
        self.idx2Sym = { v:k for k, v in self.sym2Idx.items()}
        self.dict_size = len(self.idx2Sym)
        
    def batchPadding(self, batch):
        size = max([len(record) for record in batch])
        result = np.full((len(batch), size), self.sym2Idx[self.PAD])
        for i in range(len(batch)):
            result[i][:len(batch[i])] = batch[i]
        return result

    def batchMask(self, batch):
        size = max([len(record) for record in batch])
        result = np.full((len(batch), size), 0.0)
        for i in range(len(batch)):
            result[i][:len(batch[i])] = 1.0
        return result
        
    def generator(self, batch_size, epouch):
            train = []
            length = []
            while(epouch < 0 or epouch > 0):
                for seq in self.seqs:
                    train.append([self.sym2Idx[sym] for sym in seq])
                    length.append(len(seq))
                    if(len(train) == batch_size):
                        yield self.batchPadding(train), length, self.batchMask(train)
                        train = []
                        length = []
                epouch -= 1
                print('epouch done...')
        
    def getGenerator(self, batch_size=32, epouch=-1):
        return self.generator(batch_size, epouch)
    
    def describe(self):
        print("Number of sequences: {}".format(len(self.seqs)))
        print("Longest sequence length: {}".format(self.max_length()))
        total_sym = len(self.idx2Sym)
        print("Distinct symbols: {}".format(total_sym))
        print("Top {0} most frequent symbols: {1}".format(min(total_sym, 10), [self.idx2Sym[i] for i in range(min(total_sym, 10))] ))
        print("Top {0} least frequent symbols: {1}".format(min(total_sym, 10), [self.idx2Sym[len(self.idx2Sym) - i - 1] for i in range(min(total_sym, 10))] ))
        print("Special Symbols: {}, {}, {}".format(self.GO, self.PAD, self.UNK))
        print("First batch:\n{}".format(next(self.getGenerator(3, 1))))

if __name__=='__main__':
    pass
