def build_index(iterator):
    obj2Idx = {}
    index = 0;
    for obj in iterator:
        if obj in obj2Idx.keys():
            continue
        obj2Idx[obj] = index
        index += 1

    idx2Obj = { idx: obj for obj, idx in obj2Idx.items() }

    return obj2Idx, idx2Obj

def test():
    m1, m2 = build_index(['a', 'a', 'c', 'b', 'a', 'b'])
    print(m1)
    print(m2)

if __name__=='__main__':
    test()

