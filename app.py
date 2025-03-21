import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from text_preprocessing import preprocess_text
from text_analysis import analyze_sentiment, analyze_text_with_gemini
from topic_modeling import perform_topic_modeling
from utils import get_indonesian_stopwords, display_wordcloud, apply_custom_css
import time
import os

# Set page configuration
st.set_page_config(
    page_title="Indonesian Text Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state variables if they don't exist
if 'processed_text' not in st.session_state:
    st.session_state.processed_text = None
if 'sentiment_result' not in st.session_state:
    st.session_state.sentiment_result = None
if 'topic_result' not in st.session_state:
    st.session_state.topic_result = None
if 'raw_text' not in st.session_state:
    st.session_state.raw_text = ""
if 'full_analysis' not in st.session_state:
    st.session_state.full_analysis = None
if 'api_key' not in st.session_state:
    # Get API key from environment variable or use the provided key
    st.session_state.api_key = os.getenv("GEMINI_API_KEY", "AIzaSyBJXSaPA8cPKloNc27rwcl0mNA5_NBkwuE")

# Title and description
st.title("🇮🇩 Indonesian Text Analysis")
st.markdown("""
This application analyzes Indonesian text using advanced NLP techniques and the GEMINI API.
Upload your text to get insights on sentiment, topics, and more.
""")

# Sidebar for API Key and actions
with st.sidebar:
    st.header("⚙️ Settings")
    
    api_key = st.text_input(
        "GEMINI API Key",
        value=st.session_state.api_key,
        placeholder="Enter your GEMINI API key",
        help="Enter your GEMINI API key to use for analysis",
        type="password"
    )
    
    if api_key != st.session_state.api_key:
        st.session_state.api_key = api_key
    
    st.divider()
    st.header("📋 Actions")
    
    if st.button("Clear Results", use_container_width=True):
        st.session_state.processed_text = None
        st.session_state.sentiment_result = None
        st.session_state.topic_result = None
        st.session_state.raw_text = ""
        st.session_state.full_analysis = None
        st.rerun()
    
    st.divider()
    st.markdown("### 🔍 About")
    st.info(
        """
        This application performs text analysis specifically for Indonesian language.
        
        Features:
        - Text preprocessing
        - Sentiment analysis
        - Topic modeling
        - Comprehensive text insights
        
        Powered by GEMINI API
        """
    )

# Main content area
st.header("📝 Input Text")

# Text input options
input_option = st.radio(
    "Choose input method:",
    ["Enter text", "Upload file"],
    horizontal=True
)

if input_option == "Enter text":
    user_input = st.text_area(
        "Enter Indonesian text to analyze:",
        value=st.session_state.raw_text,
        height=200,
        placeholder="Paste your Indonesian text here...",
    )
    if user_input != st.session_state.raw_text:
        st.session_state.raw_text = user_input
else:
    uploaded_file = st.file_uploader("Upload text file", type=['txt'])
    if uploaded_file is not None:
        user_input = uploaded_file.read().decode("utf-8")
        st.session_state.raw_text = user_input
        st.text_area("File content:", value=user_input, height=200, disabled=True)
    else:
        user_input = st.session_state.raw_text

# Analysis options
st.header("🔍 Analysis Options")
col1, col2, col3 = st.columns(3)

with col1:
    do_preprocessing = st.checkbox("Text Preprocessing", value=True)
with col2:
    do_sentiment = st.checkbox("Sentiment Analysis", value=True)
with col3:
    do_topic = st.checkbox("Topic Modeling", value=True)

if not st.session_state.api_key:
    st.warning("Please enter your GEMINI API key in the sidebar to perform the analysis.")

# Process button
if st.button("Analyze Text", type="primary", disabled=not st.session_state.api_key or not user_input):
    if not user_input:
        st.error("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing text..."):
            # Store raw text
            st.session_state.raw_text = user_input
            
            # Perform preprocessing if selected
            if do_preprocessing:
                progress_bar = st.progress(0)
                
                st.info("Preprocessing text...")
                processed_text = preprocess_text(user_input)
                st.session_state.processed_text = processed_text
                progress_bar.progress(33)
                
                # Perform sentiment analysis if selected
                if do_sentiment:
                    st.info("Analyzing sentiment...")
                    sentiment_result = analyze_sentiment(user_input, st.session_state.api_key)
                    st.session_state.sentiment_result = sentiment_result
                progress_bar.progress(66)
                
                # Perform topic modeling if selected
                if do_topic:
                    st.info("Extracting topics...")
                    topic_result = perform_topic_modeling(processed_text)
                    st.session_state.topic_result = topic_result
                progress_bar.progress(100)
            
            # Perform comprehensive analysis with GEMINI
            st.info("Getting comprehensive insights from GEMINI...")
            gemini_analysis = analyze_text_with_gemini(user_input, st.session_state.api_key)
            st.session_state.full_analysis = gemini_analysis
            
            time.sleep(0.5)  # Brief pause to show completion

# Display results if available
if st.session_state.raw_text:
    st.header("📊 Analysis Results")
    
    tabs = st.tabs(["Overview", "Preprocessing", "Sentiment Analysis", "Topic Modeling", "Full Analysis"])
    
    with tabs[0]:
        st.subheader("Text Overview")
        
        # Text statistics
        col1, col2, col3, col4 = st.columns(4)
        raw_text = st.session_state.raw_text
        
        with col1:
            st.metric("Characters", len(raw_text))
        with col2:
            st.metric("Words", len(raw_text.split()))
        with col3:
            st.metric("Sentences", len([s for s in raw_text.split('.') if s.strip()]))
        with col4:
            if st.session_state.sentiment_result:
                sentiment = st.session_state.sentiment_result.get("sentiment", "Unknown")
                st.metric("Sentiment", sentiment)
        
        # Summary
        if st.session_state.full_analysis:
            with st.expander("Text Summary", expanded=True):
                summary = st.session_state.full_analysis.get("summary", "No summary available.")
                st.write(summary)
        
        # Quick insights
        if st.session_state.full_analysis and "key_points" in st.session_state.full_analysis:
            st.subheader("Key Insights")
            key_points = st.session_state.full_analysis.get("key_points", [])
            for i, point in enumerate(key_points):
                st.markdown(f"**{i+1}.** {point}")
        
    with tabs[1]:
        st.subheader("Text Preprocessing")
        
        if st.session_state.processed_text:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Original Text")
                st.text_area("", value=st.session_state.raw_text, height=200, disabled=True)
            
            with col2:
                st.markdown("#### Processed Text")
                processed_text_str = " ".join(st.session_state.processed_text)
                st.text_area("", value=processed_text_str, height=200, disabled=True)
            
            # Show preprocessing steps
            with st.expander("Preprocessing Steps", expanded=True):
                st.markdown("""
                The following steps were applied during preprocessing:
                1. **Case normalization**: Convert text to lowercase
                2. **Tokenization**: Split text into words
                3. **Stopword removal**: Remove common Indonesian stopwords
                4. **Punctuation removal**: Remove punctuation marks
                5. **Number removal**: Remove numeric characters
                """)
            
            # Display wordcloud
            st.markdown("#### Word Cloud")
            wordcloud_fig = display_wordcloud(st.session_state.processed_text)
            st.pyplot(wordcloud_fig)
        else:
            st.info("Run the preprocessing to see results here.")
    
    with tabs[2]:
        st.subheader("Sentiment Analysis")
        
        if st.session_state.sentiment_result:
            result = st.session_state.sentiment_result
            
            # Display sentiment with appropriate color
            sentiment = result.get("sentiment", "Neutral")
            sentiment_color = {
                "Positive": "green",
                "Negative": "red",
                "Neutral": "gray"
            }.get(sentiment, "blue")
            
            st.markdown(f"### Overall Sentiment: <span style='color:{sentiment_color}'>{sentiment}</span>", unsafe_allow_html=True)
            
            # Display confidence scores with gauge charts
            st.subheader("Sentiment Scores")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                positive_score = result.get("positive_score", 0) * 100
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=positive_score,
                    title={'text': "Positive"},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "green"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 100], 'color': "lightgreen"}
                        ]
                    }
                ))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                neutral_score = result.get("neutral_score", 0) * 100
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=neutral_score,
                    title={'text': "Neutral"},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "gray"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 100], 'color': "silver"}
                        ]
                    }
                ))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
            
            with col3:
                negative_score = result.get("negative_score", 0) * 100
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=negative_score,
                    title={'text': "Negative"},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "red"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 100], 'color': "lightcoral"}
                        ]
                    }
                ))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
            
            # Display sentiment explanation if available
            if "explanation" in result:
                with st.expander("Sentiment Analysis Details", expanded=True):
                    st.write(result["explanation"])
        else:
            st.info("Run the sentiment analysis to see results here.")
    
    with tabs[3]:
        st.subheader("Topic Modeling")
        
        if st.session_state.topic_result:
            topics = st.session_state.topic_result
            
            # Display topics
            st.markdown("### Discovered Topics")
            
            for i, topic in enumerate(topics):
                with st.expander(f"Topic {i+1}", expanded=i==0):
                    # Display words and weights
                    topic_df = pd.DataFrame(topic, columns=["Word", "Weight"])
                    
                    col1, col2 = st.columns([2, 3])
                    
                    with col1:
                        st.dataframe(topic_df, hide_index=True, use_container_width=True)
                    
                    with col2:
                        # Create horizontal bar chart
                        fig = px.bar(
                            topic_df,
                            x="Weight",
                            y="Word",
                            orientation='h',
                            color="Weight",
                            color_continuous_scale="Viridis",
                            title=f"Word Importance in Topic {i+1}"
                        )
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
            
            # Topic distribution if available
            if st.session_state.full_analysis and "topic_distribution" in st.session_state.full_analysis:
                st.subheader("Topic Distribution")
                topic_dist = st.session_state.full_analysis["topic_distribution"]
                topic_df = pd.DataFrame(topic_dist)
                fig = px.pie(
                    topic_df, 
                    values='percentage', 
                    names='topic', 
                    title='Topic Distribution in Text',
                    color_discrete_sequence=px.colors.qualitative.Plotly
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run the topic modeling to see results here.")
    
    with tabs[4]:
        st.subheader("Full Text Analysis")
        
        if st.session_state.full_analysis:
            analysis = st.session_state.full_analysis
            
            # Display comprehensive analysis
            if "comprehensive_analysis" in analysis:
                st.markdown("### Comprehensive Analysis")
                st.write(analysis["comprehensive_analysis"])
            
            # Display language characteristics if available
            if "language_characteristics" in analysis:
                st.markdown("### Language Characteristics")
                st.write(analysis["language_characteristics"])
            
            # Display recommendations if available
            if "recommendations" in analysis:
                st.markdown("### Recommendations")
                for i, rec in enumerate(analysis["recommendations"]):
                    st.markdown(f"**{i+1}.** {rec}")
        else:
            st.info("Complete the analysis to see comprehensive results here.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Made with ❤️ for Indonesian language analysis | Powered by GEMINI API</p>
    </div>
    """, 
    unsafe_allow_html=True
)
