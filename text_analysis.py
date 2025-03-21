import google.generativeai as genai
import json

def analyze_sentiment(text, api_key):
    """
    Analyze the sentiment of Indonesian text using GEMINI API.
    
    Args:
        text (str): The Indonesian text to analyze
        api_key (str): GEMINI API key
        
    Returns:
        dict: Sentiment analysis results
    """
    try:
        # Configure the Google Generative AI with the API key
        genai.configure(api_key=api_key)
        
        # Set up the model
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Create the prompt for sentiment analysis
        prompt = f"""
        Analyze the sentiment of the following Indonesian text and provide a detailed explanation. 
        The analysis should include:
        1. Overall sentiment (Positive, Negative, or Neutral)
        2. Confidence scores for each sentiment category (as decimal numbers between 0 and 1)
        3. Brief explanation of the sentiment analysis
        
        Format the response as valid JSON with the following structure:
        {{
            "sentiment": "Positive/Negative/Neutral",
            "positive_score": 0.X,
            "neutral_score": 0.X,
            "negative_score": 0.X,
            "explanation": "Brief explanation here"
        }}
        
        Text to analyze:
        {text}
        
        Only respond with the JSON, no other text.
        """
        
        # Generate the response
        response = model.generate_content(prompt)
        
        # Extract and parse the JSON response
        response_text = response.text
        
        # Parse the JSON
        sentiment_result = json.loads(response_text)
        
        return sentiment_result
    
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        # Return a default response in case of error
        return {
            "sentiment": "Neutral",
            "positive_score": 0.33,
            "neutral_score": 0.34,
            "negative_score": 0.33,
            "explanation": f"Error analyzing sentiment: {str(e)}"
        }

def analyze_text_with_gemini(text, api_key):
    """
    Perform comprehensive text analysis using GEMINI API.
    
    Args:
        text (str): The Indonesian text to analyze
        api_key (str): GEMINI API key
        
    Returns:
        dict: Comprehensive analysis results
    """
    try:
        # Configure the Google Generative AI with the API key
        genai.configure(api_key=api_key)
        
        # Set up the model
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Create the prompt for comprehensive analysis
        prompt = f"""
        Perform a comprehensive analysis of the following Indonesian text. 
        The analysis should include:
        
        1. A summary of the text (2-3 sentences)
        2. Key points or main ideas (list format)
        3. Identified topics and their relevance (%)
        4. Language characteristics specific to Indonesian
        5. Comprehensive analysis of the content, style, and tone
        6. Recommendations for improving the text (if applicable)
        
        Format the response as valid JSON with the following structure:
        {{
            "summary": "Brief summary here",
            "key_points": ["Point 1", "Point 2", "Point 3"],
            "topic_distribution": [
                {{"topic": "Topic 1", "percentage": X}},
                {{"topic": "Topic 2", "percentage": Y}}
            ],
            "language_characteristics": "Analysis of Indonesian language characteristics",
            "comprehensive_analysis": "Detailed analysis of content, style, and tone",
            "recommendations": ["Recommendation 1", "Recommendation 2"]
        }}
        
        Text to analyze:
        {text}
        
        Only respond with the JSON, no other text.
        """
        
        # Generate the response
        response = model.generate_content(prompt)
        
        # Extract and parse the JSON response
        response_text = response.text
        
        # Parse the JSON
        analysis_result = json.loads(response_text)
        
        return analysis_result
    
    except Exception as e:
        print(f"Error in comprehensive analysis: {e}")
        # Return a default response in case of error
        return {
            "summary": f"Error analyzing text: {str(e)}",
            "key_points": ["Could not extract key points due to an error"],
            "topic_distribution": [
                {"topic": "Unknown", "percentage": 100}
            ],
            "language_characteristics": "Could not analyze language characteristics",
            "comprehensive_analysis": "Analysis failed due to an error",
            "recommendations": ["Try again with a different text"]
        }
