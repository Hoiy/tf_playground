import matplotlib.pyplot as plt

def hist(data, bin_width=1, title='Data Histogram'):
    plt.hist(data, range(min(data), max(data) + bin_width, bin_width))
    plt.title(title)
    plt.show()

def tally(iteratable):
    from collections import defaultdict
    count = defaultdict(int)
    for i in iteratable:
        count[i] = count[i] + 1
    return count

def test():
    hist([1])
    tally([['a', 'b'], ['a', 'b'], ['c', 'd']])

if __name__=='__main__':
    test()
