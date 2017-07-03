import numpy as np

# build index, i.e. sym2Idx, idx2sym, idx2Emb
# index 0 is preseved for zero_mask when used with pad_sequence
# sliming the embedding by removing embedding that does not occurs
# Add <UNK> symbol
# final embedding is np.array format
def compact_embedding(embedding, symbols):
    assert isinstance(embedding, dict)
    assert isinstance(symbols, list)

    compact_emb = [[0. for v in next(iter(embedding.values()))]]
    sym2Idx = {}
    index = 1
    matched = 0
    for sym in symbols:
        if sym in embedding:
            matched = matched + 1
            if sym not in sym2Idx:
                sym2Idx[sym] = index
                compact_emb.append(embedding[sym])
                index = index + 1
    idx2Sym = { idx: sym for sym, idx in sym2Idx.items() }

    print("Size of compact embedding: {0}".format(len(compact_emb)))
    print("Embedding coverage: {0:.2f}%".format(matched / len(symbols) * 100.))

    emb = np.array(compact_emb)

    def s2i(symbols):
        return [sym2Idx[sym] if sym in sym2Idx else 0 for sym in symbols]

    def i2s(indice):
        return [idx2Sym[idx] if idx > 0 else '<UNK>' for idx in indice]

    def i2v(indice):
        return np.array([emb[idx] for idx in indice])

    return emb, s2i, i2s, i2v




def test():
    corpus = ['the', 'dog', 'barks', '.']
    embedding = {
        '.': [0,1,2],
        'the': [1,2,3],
        'dog': [2,3,4],
        'cat': [3,4,5],
        'a': [4,5,6]
    }
    emb, s2i, i2s, i2v = compact_embedding(embedding, corpus)
    print(emb)
    indice = s2i(corpus)
    print(s2i(corpus))
    print(i2s(indice))
    print(i2v(indice))


if __name__=='__main__':
    test()
