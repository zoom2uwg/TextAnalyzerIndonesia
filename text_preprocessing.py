import re
import string
from nltk.tokenize import word_tokenize
from utils import get_indonesian_stopwords

def preprocess_text(text):
    """
    Preprocess Indonesian text for analysis.
    
    Args:
        text (str): Raw Indonesian text
        
    Returns:
        list: List of preprocessed tokens
    """
    try:
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S*@\S*\s?', '', text)
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Get stopwords
        stop_words = get_indonesian_stopwords()
        
        # Remove stopwords
        tokens = [token for token in tokens if token not in stop_words and len(token) > 1]
        
        return tokens
    
    except Exception as e:
        # In case of any errors, return a simplified preprocessing
        print(f"Error in preprocessing: {e}")
        
        # Simple fallback tokenization
        text = text.lower()
        text = ''.join([c for c in text if c not in string.punctuation])
        tokens = text.split()
        return [t for t in tokens if len(t) > 1]

def extract_sentences(text):
    """
    Extract sentences from text.
    
    Args:
        text (str): Raw text
        
    Returns:
        list: List of sentences
    """
    # Simple sentence splitting by punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Clean up and remove empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences

def count_words(text):
    """
    Count the number of words in text.
    
    Args:
        text (str): Raw text
        
    Returns:
        int: Number of words
    """
    # Simple word counting
    words = re.findall(r'\b\w+\b', text.lower())
    return len(words)

def get_word_frequency(tokens, top_n=20):
    """
    Get word frequency from tokens.
    
    Args:
        tokens (list): List of tokens
        top_n (int): Number of top words to return
        
    Returns:
        dict: Dictionary of word frequencies
    """
    # Count word frequencies
    word_freq = {}
    for token in tokens:
        word_freq[token] = word_freq.get(token, 0) + 1
    
    # Sort by frequency
    sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    # Return top N words
    return dict(sorted_word_freq[:top_n])
