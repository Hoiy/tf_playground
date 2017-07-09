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


# Use case, random spllit data into training and validation set
# input window size, data, output training generator, testing generator

def sliding_window_random_access(data, window_size, validation_size=0.2):
    assert isinstance(data, str) 
    assert window_size >= 1
   
    indice = list(range(len(data)-window_size + 1))
    import random
    random.shuffle(indice)
    split_idx = int(len(indice) * (1.-validation_size))
    idx = {}
    idx['train'] = indice[:split_idx]
    idx['test'] = indice[split_idx:]
    def generator(mode):
        while True:
            random.shuffle(idx[mode])
            for i in idx[mode]:
                yield data[i:i+window_size]
    print('Training data size:', len(idx['train']))
    print('Testing data size:', len(idx['test']))
    return generator('train'), generator('test'), len(idx['train']), len(idx['test'])

def test():
    for j in range(1, 5):
        generator = sliding_window(j)(['a', 'b', 'c', 'd'])
        print(j)
        for i in range(5):
            print(i)

        generator = random_window(j)(['a', 'b', 'c', 'd'])
        print(j)
        for i in range(5):
            print(next(generator))

    generator = transform(sliding_window(2)(['a', 'b', 'c', 'd']), lambda x: x[0] + x[1])
    for i in range(5):
        print(i)

    train_gen, test_gen, train_size, test_size = sliding_window_random_access('abcdefg', 3)
    for i in range(10):
        print(next(train_gen))
    for i in range(10):
        print(next(test_gen))


if __name__=='__main__':
    test()
