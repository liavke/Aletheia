import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import unicodedata
import re
import string
from tensorflow.keras.preprocessing.text import text_to_word_sequence

class worldPreProcessor():
    def __init__(self):
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('stopwords')
        return self

    def remove_accented_chars(text):
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return text

    def remove_special_characters(text):
        pattern = r'[^a-zA-Z\s]'
        n_pattern = r'[^a-zA-Z0-9\s]'
        # Removing everything apart from alphanumerical chars
        text = re.sub(pattern, '', text)
        # Removing numbers
        text = re.sub(n_pattern, '', text)
        return text

    def to_lower(text):
        return text.lower().strip()

    def remove_p(text):
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub('[''""…]', '', text)
        text = re.sub('\n', '', text)
        return text

    def tokenization(text):
        """
         If the input is a string (sentence) the output will be the list of words/tokens in that sentence.
        """
        tokens = nltk.word_tokenize(text)
        return tokens

    def remove_stopwords(tokens):
        """
        Stopwords are the words which are most common like I, am, there, where etc
        """
        filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
        return filtered_tokens

    def stem(words):
        """
        reducing a given token/word to its root form
        """
        ps = PorterStemmer()
        stemmed_tokens = [ps.stem(word) for word in words]
        return stemmed_tokens

    def lemmatize(words):
        """
        transforms to the actual root word based on a dictionary
        :return: List of dictionary defined words
        """
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(word) for word in words]
        return lemmatized_tokens
