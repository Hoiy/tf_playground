from collections import Counter

# sorted by desc frequency as nce_loss function use uni-log sampling
def convert(words):
    counter = Counter(words)
    word_dict = {}
    index = 0
    for item in sorted(counter.items(), key=lambda x: (x[1], x[0]), reverse=True):
        word_dict[item[0]] = index
        index += 1
     
    return word_dict, { v:k for k, v in word_dict.items()}


if __name__=='__main__':
    pass
