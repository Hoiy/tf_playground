def sliding_window(size):
    assert size > 0
    def generator(iteratable):
        while True:
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

def transform(generator, transformer):
    for i in generator:
        yield transformer(i)

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

    generator = transform(sliding_window(2)(['a', 'b', 'c', 'd']), lambda x: x[0] + x[1])
    for i in generator:
        print(i)


if __name__=='__main__':
    test()
