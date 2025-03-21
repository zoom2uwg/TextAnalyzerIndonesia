import string
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import io

def get_indonesian_stopwords():
    """
    Get a list of Indonesian stopwords.
    
    Returns:
        set: Set of Indonesian stopwords
    """
    # Common Indonesian stopwords
    stopwords = {
        'yang', 'dan', 'di', 'dengan', 'untuk', 'tidak', 'ini', 'dari', 'dalam', 
        'akan', 'pada', 'juga', 'ke', 'karena', 'ada', 'saya', 'kamu', 'kita', 
        'itu', 'atau', 'oleh', 'bisa', 'saat', 'para', 'tahun', 'adalah', 'bahwa',
        'mereka', 'dia', 'saat', 'sedang', 'lebih', 'kami', 'telah', 'dengan',
        'sudah', 'apa', 'tersebut', 'masih', 'jika', 'maka', 'sebagai', 'belum',
        'ayo', 'yuk', 'apakah', 'ya', 'ya', 'nya', 'lah', 'pun', 'dong', 'kan',
        'kah', 'deh', 'tuh', 'yah', 'nih', 'sih', 'hi', 'ku', 'mu', 'nya',
        'jangan', 'nggak', 'gak', 'tak', 'enggak', 'ndak',
        'si', 'eh', 'ah', 'oh', 'hm', 'hmm', 'wah', 'loh', 'kok'
    }
    
    # Add more variations of each stopword
    expanded_stopwords = set(stopwords)
    for word in stopwords:
        # Add common variations with punctuation
        expanded_stopwords.add(word + '.')
        expanded_stopwords.add(word + ',')
        expanded_stopwords.add(word + ':')
        expanded_stopwords.add(word + ';')
        expanded_stopwords.add(word + '?')
        expanded_stopwords.add(word + '!')
    
    return expanded_stopwords

def display_wordcloud(tokens, width=800, height=400):
    """
    Create a word cloud visualization from preprocessed tokens.
    
    Args:
        tokens (list): List of preprocessed tokens
        width (int): Width of the word cloud image
        height (int): Height of the word cloud image
        
    Returns:
        matplotlib.figure.Figure: Figure containing the word cloud
    """
    # Create word frequency dictionary
    word_freq = {}
    for token in tokens:
        word_freq[token] = word_freq.get(token, 0) + 1
    
    # Create word cloud
    wordcloud = WordCloud(
        width=width,
        height=height,
        background_color='white',
        colormap='viridis',
        max_words=100,
        collocations=False,
        contour_width=1,
        contour_color='steelblue'
    ).generate_from_frequencies(word_freq)
    
    # Create a figure for the word cloud
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    plt.tight_layout(pad=0)
    
    return fig

def truncate_text(text, max_length=100):
    """
    Truncate text to a maximum length.
    
    Args:
        text (str): Text to truncate
        max_length (int): Maximum length
        
    Returns:
        str: Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."

def apply_custom_css():
    """
    Apply custom CSS to the Streamlit app.
    
    Returns:
        str: CSS string
    """
    # This is a placeholder function in case we need to add custom CSS in the future.
    # For now, we're using Streamlit's default styling as per the requirements.
    css = """
    <style>
    </style>
    """
    return css

def format_json_for_display(json_data):
    """
    Format JSON data for display.
    
    Args:
        json_data (dict): JSON data
        
    Returns:
        str: Formatted JSON string
    """
    import json
    
    # Convert to a formatted string
    formatted_json = json.dumps(json_data, indent=2)
    
    return formatted_json
