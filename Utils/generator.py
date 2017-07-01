def sliding_window(size):
    assert size > 0
    def sliding_window_generator(iteratable):
        for i in range(len(iteratable) - size + 1):
            yield iteratable[i:i+size]
    return sliding_window_generator

def test():
    for j in range(1, 5):
        generator = sliding_window(j)(['a', 'b', 'c', 'd'])
        print(j)
        for i in generator:
            print(i)

if __name__=='__main__':
    test()
