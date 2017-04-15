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

def txt_read(path):
    try:
        with open(path, 'r') as f:
            return f.read()
    except:
        print('Failed to load ' + path)
        return ""

def txt_write(path, data):
    with open(path, 'w') as f:
        f.write(data)

def ls(path):
    path = os.path.abspath(path)
    return [os.path.join(path, f) for f in os.listdir(path)]

def test():
   obj = [1, 2]
   write('/tmp/test.pkl', obj)
   print(read('/tmp/test.pkl'))

def test2():
   files = ls('./data/Gutenberg')
   corpus = ''
   for f in files:
       corpus += read(f)

if __name__=='__main__':
    test2()
