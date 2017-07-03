import sys

"""
Credit: http://stackoverflow.com/questions/35092720/verbosity-pandas-apply
"""

def progress_coroutine(print_on = 1000):
    print("Starting progress monitor")
    iterations = 0
    while True:
        yield
        iterations += 1
        if (iterations % print_on == 0):
            sys.stdout.write("\r{0} finished".format(iterations))
            sys.stdout.flush()

def percentage_coroutine(to_process, print_on_percent = 0.01):
    print("Starting progress percentage monitor")

    processed = 0
    count = 0
    print_count = to_process*print_on_percent
    while True:
        yield
        processed += 1
        count += 1
        if (count >= print_count):
            count = 0
            pct = (float(processed)/float(to_process))*100
            sys.stdout.write("\r{0}% finished\n".format(pct))
            sys.stdout.flush()

def trace_progress(func):
    monitor = progress_coroutine()
    monitor.send(None)
    def callf(*args, **kwargs):
        res = func(*args, **kwargs)
        monitor.send(None)
        return res

    return callf

def batch(func, batch_data):
    import collections
    assert isinstance(batch_data, collections.Iterable)
    return [func(data) for data in batch_data]

def test():
    func = trace_progress(lambda x: x)
    for i in range(1000000):
        func(i)

if __name__=='__main__':
    test()
