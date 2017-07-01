import reprlib
import types

def dump(data, max_size=250):
    r = reprlib.Repr()
    if isinstance(data, str):
        r.maxstring = max_size
    elif isinstance(data, types.GeneratorType):
        count = 0;
        while count < max_size:
            print(next(data))
            count = count + 1
        return

    elif isinstance(data, (list, tuple, dict, set)):
        r.maxdict = max_size
        r.maxlist = max_size
        r.maxtuple = max_size
        r.maxset = max_size
    else:
        r.maxlevel = max_size
        r.maxdict = max_size
        r.maxlist = max_size
        r.maxtuple = max_size
        r.maxset = max_size
        r.maxfrozenset = max_size
        r.maxdeque = max_size
        r.maxarray = max_size
        r.maxlong = max_size
        r.maxstring = max_size
        r.maxother = max_size

    print(type(data))
    print(r.repr(data))

def test():
    dump('aaaaaa');
    dump([['a', 'b', 'c'], ['a', 'b', 'c'], ['a', 'b', 'c']])

if __name__=='__main__':
    test()
