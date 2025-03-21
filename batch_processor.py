"""
Batch processor for handling multiple documents in Indonesian text analysis
"""

import concurrent.futures
import pandas as pd
import time
import logging
from text_preprocessing import preprocess_text, count_words
from text_analysis import analyze_sentiment
from topic_modeling import perform_topic_modeling
from database import save_analysis

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_text(text, text_type, api_key, process_config=None):
    """
    Process a single text with specified analysis options
    
    Args:
        text (str): The text to analyze
        text_type (str): Type of text (e.g., news, review)
        api_key (str): GEMINI API key for analysis
        process_config (dict): Configuration settings for processing options
            - do_preprocessing (bool): Whether to do preprocessing
            - do_sentiment (bool): Whether to do sentiment analysis
            - do_topic (bool): Whether to do topic modeling
            
    Returns:
        dict: Results of processing
    """
    if process_config is None:
        process_config = {
            'do_preprocessing': True, 
            'do_sentiment': True,
            'do_topic': True
        }
    
    result = {
        'text': text,
        'text_type': text_type,
        'word_count': count_words(text),
        'status': 'completed',
        'errors': []
    }
    
    try:
        # Preprocessing
        if process_config.get('do_preprocessing', True):
            processed_text = preprocess_text(text)
            result['processed_text'] = processed_text
        else:
            processed_text = None
            
        # Sentiment analysis
        if process_config.get('do_sentiment', True):
            try:
                sentiment_result = analyze_sentiment(text, api_key)
                result['sentiment_result'] = sentiment_result
            except Exception as e:
                logger.error(f"Error in sentiment analysis: {str(e)}")
                result['errors'].append(f"Sentiment analysis failed: {str(e)}")
                result['sentiment_result'] = None
        
        # Topic modeling
        if process_config.get('do_topic', True) and processed_text:
            try:
                topic_result = perform_topic_modeling(processed_text)
                result['topic_result'] = topic_result
            except Exception as e:
                logger.error(f"Error in topic modeling: {str(e)}")
                result['errors'].append(f"Topic modeling failed: {str(e)}")
                result['topic_result'] = None
                
    except Exception as e:
        logger.error(f"Error processing text: {str(e)}")
        result['status'] = 'failed'
        result['errors'].append(f"Processing failed: {str(e)}")
    
    return result

def batch_process(texts, text_types, api_key, process_config=None, save_to_db=False, max_workers=3):
    """
    Process multiple texts in batch mode
    
    Args:
        texts (list): List of texts to process
        text_types (list): List of text types corresponding to each text
        api_key (str): GEMINI API key for analysis
        process_config (dict): Configuration for processing options
        save_to_db (bool): Whether to save results to database
        max_workers (int): Maximum number of concurrent workers
        
    Returns:
        list: List of processing results
    """
    results = []
    start_time = time.time()
    
    # Process texts in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_text = {
            executor.submit(
                process_text, 
                text, 
                text_type, 
                api_key, 
                process_config
            ): (i, text, text_type) 
            for i, (text, text_type) in enumerate(zip(texts, text_types))
        }
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_text):
            idx, text, text_type = future_to_text[future]
            try:
                result = future.result()
                results.append(result)
                
                # Save to database if requested
                if save_to_db and result['status'] == 'completed':
                    try:
                        preprocessed_text_str = " ".join(result.get('processed_text', [])) if result.get('processed_text') else None
                        
                        record_id = save_analysis(
                            text=text,
                            text_type=text_type,
                            word_count=result.get('word_count'),
                            preprocessed_text=preprocessed_text_str,
                            sentiment_analysis=result.get('sentiment_result'),
                            topic_modeling=result.get('topic_result')
                        )
                        
                        result['record_id'] = record_id
                    except Exception as e:
                        logger.error(f"Error saving to database: {str(e)}")
                        result['errors'].append(f"Database save failed: {str(e)}")
                
            except Exception as e:
                logger.error(f"Error processing text {idx}: {str(e)}")
                results.append({
                    'text': text[:100] + "..." if len(text) > 100 else text,
                    'text_type': text_type,
                    'status': 'failed',
                    'errors': [str(e)]
                })
    
    # Sort results by original order
    sorted_results = sorted(results, key=lambda x: texts.index(x['text']) if x['text'] in texts else 0)
    
    execution_time = time.time() - start_time
    logger.info(f"Batch processing completed in {execution_time:.2f} seconds")
    
    return sorted_results

def generate_batch_report(results):
    """
    Generate a summary report of batch processing results
    
    Args:
        results (list): List of processing results
        
    Returns:
        dict: Report with summary statistics and comparison
    """
    report = {
        'total_texts': len(results),
        'successful': len([r for r in results if r['status'] == 'completed']),
        'failed': len([r for r in results if r['status'] == 'failed']),
        'sentiment_distribution': {},
        'word_count_stats': {},
        'topic_summary': []
    }
    
    # Only process successful results for stats
    successful_results = [r for r in results if r['status'] == 'completed']
    
    # Sentiment distribution
    sentiments = []
    for result in successful_results:
        if result.get('sentiment_result'):
            sentiment = result['sentiment_result'].get('sentiment', 'Unknown')
            sentiments.append(sentiment)
    
    if sentiments:
        sentiment_counts = pd.Series(sentiments).value_counts().to_dict()
        report['sentiment_distribution'] = sentiment_counts
    
    # Word count statistics
    word_counts = [r.get('word_count', 0) for r in successful_results]
    if word_counts:
        report['word_count_stats'] = {
            'min': min(word_counts),
            'max': max(word_counts),
            'avg': sum(word_counts) / len(word_counts),
            'total': sum(word_counts)
        }
    
    # Topic summary (collect top words across all texts)
    all_topic_words = []
    for result in successful_results:
        if result.get('topic_result'):
            for topic in result['topic_result']:
                all_topic_words.extend([word for word, _ in topic[:5]])  # Add top 5 words from each topic
    
    if all_topic_words:
        # Get most common words across all topics
        word_counts = pd.Series(all_topic_words).value_counts().head(20).to_dict()
        report['topic_summary'] = [{'word': word, 'count': count} for word, count in word_counts.items()]
    
    return report

def compare_texts(results):
    """
    Compare multiple texts based on their analysis results
    
    Args:
        results (list): List of processing results
        
    Returns:
        dict: Comparison results
    """
    # Filter for successful results only
    successful_results = [r for r in results if r['status'] == 'completed']
    
    if len(successful_results) < 2:
        return {"error": "Need at least 2 successful analyses to compare"}
    
    comparison = {
        'sentiment_comparison': [],
        'length_comparison': [],
        'topic_similarity': []
    }
    
    # Compare sentiments
    for i, result in enumerate(successful_results):
        if result.get('sentiment_result'):
            comparison['sentiment_comparison'].append({
                'text_num': i + 1,
                'text_type': result.get('text_type', 'Unknown'),
                'sentiment': result['sentiment_result'].get('sentiment', 'Unknown'),
                'positive_score': result['sentiment_result'].get('positive_score', 0),
                'neutral_score': result['sentiment_result'].get('neutral_score', 0),
                'negative_score': result['sentiment_result'].get('negative_score', 0)
            })
    
    # Compare text lengths
    for i, result in enumerate(successful_results):
        comparison['length_comparison'].append({
            'text_num': i + 1,
            'text_type': result.get('text_type', 'Unknown'),
            'word_count': result.get('word_count', 0),
            'char_count': len(result['text'])
        })
    
    # Calculate topic similarity between texts
    # (Basic implementation using overlap of top topic words)
    for i in range(len(successful_results)):
        for j in range(i+1, len(successful_results)):
            r1 = successful_results[i]
            r2 = successful_results[j]
            
            if r1.get('topic_result') and r2.get('topic_result'):
                # Extract top words from each text's topics
                words1 = set()
                words2 = set()
                
                for topic in r1['topic_result']:
                    words1.update([word for word, _ in topic[:10]])  # Top 10 words
                
                for topic in r2['topic_result']:
                    words2.update([word for word, _ in topic[:10]])  # Top 10 words
                
                # Calculate Jaccard similarity (intersection over union)
                intersection = len(words1.intersection(words2))
                union = len(words1.union(words2))
                similarity = intersection / union if union > 0 else 0
                
                comparison['topic_similarity'].append({
                    'text_pair': f"Text {i+1} & Text {j+1}",
                    'types': f"{r1.get('text_type', 'Unknown')} & {r2.get('text_type', 'Unknown')}",
                    'similarity_score': similarity,
                    'common_words': list(words1.intersection(words2))
                })
    
    return comparison