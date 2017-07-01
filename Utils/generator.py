def sliding_window(size):
    assert size > 0
    def generator(iteratable):
        for i in range(len(iteratable) - size + 1):
            yield iteratable[i:i+size]
    return generator

def random_window(size):
    assert size > 0
    def generator(iteratable):
        import random
        while(True):
            start = random.randint(0, len(iteratable) - size)
            yield iteratable[start:start+size]
    return generator

def test():
    for j in range(1, 5):
        generator = sliding_window(j)(['a', 'b', 'c', 'd'])
        print(j)
        for i in generator:
            print(i)

        generator = random_window(j)(['a', 'b', 'c', 'd'])
        print(j)
        for i in range(5):
            print(next(generator))

if __name__=='__main__':
    test()
