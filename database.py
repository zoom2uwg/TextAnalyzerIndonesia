import os
import json
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get database URL from environment variables
DATABASE_URL = os.getenv('DATABASE_URL')

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)

# Create base class for models
Base = declarative_base()

# Define the TextAnalysis model
class TextAnalysis(Base):
    __tablename__ = 'text_analysis'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    text = Column(Text)
    text_type = Column(String(50), nullable=True)
    word_count = Column(Integer, nullable=True)
    preprocessed_text = Column(Text, nullable=True)
    sentiment_analysis = Column(JSON, nullable=True)
    topic_modeling = Column(JSON, nullable=True)
    
    def __repr__(self):
        return f"<TextAnalysis(id={self.id}, timestamp={self.timestamp})>"

# Create tables
Base.metadata.create_all(engine)

# Create session
Session = sessionmaker(bind=engine)

def save_analysis(text, text_type=None, word_count=None, preprocessed_text=None, 
                 sentiment_analysis=None, topic_modeling=None):
    """
    Save text analysis results to the database.
    
    Args:
        text (str): Original text
        text_type (str, optional): Type of text
        word_count (int, optional): Word count
        preprocessed_text (str, optional): Preprocessed text
        sentiment_analysis (dict, optional): Sentiment analysis results
        topic_modeling (dict, optional): Topic modeling results
        
    Returns:
        int: ID of the saved record
    """
    try:
        session = Session()
        
        # Convert dict to JSON if needed
        sentiment_json = json.dumps(sentiment_analysis) if sentiment_analysis else None
        topic_json = json.dumps(topic_modeling) if topic_modeling else None
        
        analysis = TextAnalysis(
            text=text,
            text_type=text_type,
            word_count=word_count,
            preprocessed_text=preprocessed_text,
            sentiment_analysis=sentiment_json,
            topic_modeling=topic_json
        )
        
        session.add(analysis)
        session.commit()
        
        record_id = analysis.id
        session.close()
        
        return record_id
    except Exception as e:
        print(f"Error saving to database: {e}")
        return None

def get_analysis_history(limit=10):
    """
    Get text analysis history from the database.
    
    Args:
        limit (int): Maximum number of records to retrieve
        
    Returns:
        list: List of analysis records
    """
    try:
        session = Session()
        
        results = session.query(TextAnalysis).order_by(
            TextAnalysis.timestamp.desc()
        ).limit(limit).all()
        
        history = []
        for result in results:
            history.append({
                'id': result.id,
                'timestamp': result.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'text': result.text[:100] + '...' if len(result.text) > 100 else result.text,
                'text_type': result.text_type,
                'word_count': result.word_count,
                'has_sentiment': result.sentiment_analysis is not None,
                'has_topics': result.topic_modeling is not None
            })
        
        session.close()
        
        return history
    except Exception as e:
        print(f"Error retrieving history from database: {e}")
        return []

def get_analysis_by_id(analysis_id):
    """
    Get a specific analysis record by ID.
    
    Args:
        analysis_id (int): ID of the analysis record
        
    Returns:
        dict: Analysis record
    """
    try:
        session = Session()
        
        result = session.query(TextAnalysis).filter(TextAnalysis.id == analysis_id).first()
        
        if result:
            analysis = {
                'id': result.id,
                'timestamp': result.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'text': result.text,
                'text_type': result.text_type,
                'word_count': result.word_count,
                'preprocessed_text': result.preprocessed_text,
                'sentiment_analysis': json.loads(result.sentiment_analysis) if result.sentiment_analysis else None,
                'topic_modeling': json.loads(result.topic_modeling) if result.topic_modeling else None
            }
        else:
            analysis = None
        
        session.close()
        
        return analysis
    except Exception as e:
        print(f"Error retrieving analysis from database: {e}")
        return None