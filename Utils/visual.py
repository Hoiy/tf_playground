import matplotlib.pyplot as plt

def hist(data, bin_width=1, title='Data Histogram'):
    plt.hist(data, range(min(data), max(data) + bin_width, bin_width))
    plt.title(title)
    plt.show()

def test():
    hist([1])

if __name__=='__main__':
    test()
