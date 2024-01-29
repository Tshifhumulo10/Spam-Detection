
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def remove_stopwords(text):
    """
    Remove stopwords from the given text.

    Args:
        text (str): The input text from which stopwords are to be removed.

    Returns:
        str: The text with stopwords removed.
    """
    stop_words = set(stopwords.words('english'))   # Set of stopwords in English
    tokens = text.split()                          # Splitting the text into individual words
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]  # Filtering out stopwords
    return ' '.join(filtered_tokens)   


def remove_punctuation(post):
    """
    Remove punctuation marks from the given post.

    Args:
        post (str): The input post from which punctuation marks are to be removed.

    Returns:
        str: The post with punctuation marks removed.
    """
    return ''.join([l for l in post if l not in string.punctuation])

def lemmatize_text(text):
    # Initializing the WordNet Lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Tokenizing the text into individual words
    words = nltk.word_tokenize(text)
    
    # Lemmatize each word and join them back into a sentence
    lemmatized_text = ' '.join(lemmatizer.lemmatize(word) for word in words)
    
    return lemmatized_text