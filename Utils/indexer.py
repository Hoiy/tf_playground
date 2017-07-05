def build_index(iterator):
    obj2Idx = {}
    index = 0;
    for obj in iterator:
        if obj in obj2Idx.keys():
            continue
        obj2Idx[obj] = index
        index += 1

    idx2Obj = { idx: obj for obj, idx in obj2Idx.items() }

    def o2i(o):
        return obj2Idx[o]

    def i2o(i):
        return idx2Obj[i]

    return o2i, i2o, len(obj2Idx)

def index_2_one_hot(index, max_size=87):
    return [1 if i == index else 0 for i in range(max_size)]

def test():
    m1, m2 = build_index(['a', 'a', 'c', 'b', 'a', 'b'])
    print(m1)
    print(m2)

if __name__=='__main__':
    test()
