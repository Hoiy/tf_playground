from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import re
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer('english')

def clean( string ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove links
    text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', string)
    #text = BeautifulSoup(string, "html5lib").get_text() 
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    #
    # 5.5 stem words
    meaningful_words = [stemmer.stem(w) for w in meaningful_words] 
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))   

def test():
    print(clean('a 1 24342 awd http://yahoo.com/test  dad good weather'))

if __name__=='__main__':
    test()
