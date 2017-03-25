import pickle
import os

def DoActionByExt(action, ext, args):
    return globals()['{0}_{1}'.format(ext, action)](*args)

def read(path):
    filename, file_ext = os.path.splitext(path)
    return DoActionByExt('read', file_ext[1:], [path])

def write(path, data):
    filename, file_ext = os.path.splitext(path)
    return DoActionByExt('write', file_ext[1:], [path, data])

def pkl_read(path):
    with open(path, 'rb') as f:
            return pickle.load(f)

def pkl_write(path, data):
    with open(path, 'wb') as f:
            return pickle.dump(data, f)

def test():
   obj = [1, 2]
   write('/tmp/test.pkl', obj)
   print(read('/tmp/test.pkl'))

if __name__=='__main__':
    test()
