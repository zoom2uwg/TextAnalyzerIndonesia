import gensim
from gensim import corpora
from gensim.models import LdaModel
import numpy as np

def perform_topic_modeling(preprocessed_tokens, num_topics=5, num_words=10):
    """
    Perform topic modeling on preprocessed text.
    
    Args:
        preprocessed_tokens (list): List of preprocessed tokens
        num_topics (int): Number of topics to extract
        num_words (int): Number of words per topic to return
        
    Returns:
        list: List of topics with their associated words and weights
    """
    try:
        # Handle case where text is too short
        if len(preprocessed_tokens) < 20:
            # Create simple topics based on word frequency
            word_freq = {}
            for token in preprocessed_tokens:
                word_freq[token] = word_freq.get(token, 0) + 1
            
            # Sort by frequency
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            
            # Create artificial topics (fallback for short text)
            topics = []
            words_per_topic = max(2, len(sorted_words) // min(num_topics, 3))
            
            for i in range(min(num_topics, 3)):
                start_idx = i * words_per_topic
                end_idx = start_idx + words_per_topic
                
                if start_idx < len(sorted_words):
                    topic_words = sorted_words[start_idx:end_idx]
                    # Normalize weights
                    total = sum(w for _, w in topic_words)
                    normalized_topic = [(word, weight/total) for word, weight in topic_words]
                    topics.append(normalized_topic)
            
            return topics
        
        # Prepare corpus for gensim
        # Create a dictionary
        dictionary = corpora.Dictionary([preprocessed_tokens])
        
        # Create a corpus
        corpus = [dictionary.doc2bow(preprocessed_tokens)]
        
        # Check if corpus and dictionary are suitable for LDA
        if not corpus or max(len(dictionary), len(corpus)) < num_topics:
            # Fallback to fewer topics
            actual_num_topics = max(2, min(len(dictionary), len(corpus)))
        else:
            actual_num_topics = num_topics
        
        # Train LDA model
        lda_model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=actual_num_topics,
            random_state=42,
            passes=10,
            alpha='auto',
            per_word_topics=True
        )
        
        # Extract topics
        topics = []
        for topic_id in range(actual_num_topics):
            topic = lda_model.show_topic(topic_id, num_words)
            topics.append(topic)
        
        return topics
    
    except Exception as e:
        print(f"Error in topic modeling: {e}")
        
        # Create a fallback topic based on word frequency
        word_freq = {}
        for token in preprocessed_tokens:
            word_freq[token] = word_freq.get(token, 0) + 1
        
        # Sort by frequency and create a single topic
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        top_words = sorted_words[:num_words]
        
        # Normalize weights
        total = sum(w for _, w in top_words)
        if total > 0:
            normalized_topic = [(word, weight/total) for word, weight in top_words]
        else:
            normalized_topic = [("no", 0.5), ("topics", 0.5)]
        
        return [normalized_topic]

def extract_keywords(preprocessed_tokens, num_keywords=10):
    """
    Extract keywords from preprocessed text using TF-IDF principles.
    
    Args:
        preprocessed_tokens (list): List of preprocessed tokens
        num_keywords (int): Number of keywords to extract
        
    Returns:
        list: List of keywords with their weights
    """
    try:
        # Count word frequencies
        word_freq = {}
        for token in preprocessed_tokens:
            word_freq[token] = word_freq.get(token, 0) + 1
        
        # Get total word count
        total_words = len(preprocessed_tokens)
        
        # Calculate term frequency (TF)
        tf = {word: count/total_words for word, count in word_freq.items()}
        
        # Sort by TF (simplified keyword extraction)
        sorted_keywords = sorted(tf.items(), key=lambda x: x[1], reverse=True)
        
        # Return top keywords
        return sorted_keywords[:num_keywords]
    
    except Exception as e:
        print(f"Error extracting keywords: {e}")
        return [("error", 1.0)]
