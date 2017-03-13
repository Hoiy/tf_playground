import pycld2 as cld2
import string
from langdetect import detect

def detect(x, package='cld2'):
    return {
        'cld2': cld2Detect,
        'langdetect': langdetectDetect
    }.get(package)(x)

def cld2Detect(input_str):
    try:
        result = cld2.detect(input_str)
    except:
        printable_str = ''.join(x for x in input_str if x in string.printable)
        try:
            result = cld2.detect(printable_str)
        except:
            return False

    if(result[0]):
        return result[2][0][1]
    else:
        return False

def langdetectDetect(input_str):
    return detect(input_str)

def test():
    print(detect('a brown fox jumps over the lazy dog'))
    print(detect('a brown fox jumps over the lazy dog', 'langdetect'))

if __name__=='__main__':
    test()
