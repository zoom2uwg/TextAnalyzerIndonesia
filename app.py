import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from text_preprocessing import preprocess_text, count_words
from text_analysis import analyze_sentiment, analyze_text_with_gemini
from topic_modeling import perform_topic_modeling
from utils import get_indonesian_stopwords, display_wordcloud, apply_custom_css, format_json_for_display
from database import save_analysis, get_analysis_history, get_analysis_by_id
from web_scraper import get_dynamic_samples
from batch_processor import batch_process, generate_batch_report, compare_texts
import time
import os
import json
import logging
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Indonesian Text Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom CSS for modern UI
st.markdown("""
<style>
    /* Modern color scheme */
    :root {
        --primary-color: #6C63FF;
        --secondary-color: #4CC9F0;
        --accent-color: #F72585;
        --background-color: #F8F9FA;
        --text-color: #212529;
    }
    
    /* Font styling */
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        color: var(--text-color);
    }
    
    /* Card-like containers */
    .stCard {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        padding: 1.5rem;
        margin-bottom: 1rem;
        background-color: white;
        transition: transform 0.3s ease;
    }
    
    .stCard:hover {
        transform: translateY(-5px);
    }
    
    /* Buttons styling */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0 0;
        padding: 10px 16px;
        font-weight: 600;
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary-color);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: var(--background-color);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: var(--primary-color);
    }
    
    /* Animation for loading states */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.3; }
        100% { opacity: 1; }
    }
    
    .stSpinner {
        animation: pulse 1.5s infinite;
    }
    
    /* Footer styling */
    footer {
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #e9ecef;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

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
if 'dynamic_samples' not in st.session_state:
    # Initialize with static samples that will be replaced with dynamic ones
    st.session_state.dynamic_samples = {}
if 'refresh_samples' not in st.session_state:
    st.session_state.refresh_samples = False

# Title and description with modern styling
st.markdown("""
<div style="text-align: center; padding: 2rem 0;">
    <h1 style="color: #6C63FF; font-size: 3rem;">🇮🇩 Indonesian Text Analysis</h1>
    <p style="font-size: 1.2rem; margin-top: 0.5rem; color: #495057;">
        Analyze Indonesian text using advanced NLP techniques powered by GEMINI API
    </p>
</div>
""", unsafe_allow_html=True)

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

# Sample Indonesian texts
sample_texts = {
    "Berita Nasional": """
    Jakarta - Pemerintah Indonesia telah mengumumkan program pembangunan infrastruktur baru yang akan dilaksanakan di seluruh wilayah Indonesia. Program ini bertujuan untuk meningkatkan konektivitas antar daerah dan mendorong pertumbuhan ekonomi. Menteri Pekerjaan Umum dan Perumahan Rakyat menyatakan bahwa proyek ini akan dimulai tahun depan dengan anggaran sebesar 500 triliun rupiah. Proyek ini diharapkan dapat mengurangi kesenjangan pembangunan antara Jawa dan luar Jawa.
    """,
    
    "Ulasan Produk": """
    Saya baru saja membeli smartphone terbaru dari merek terkenal dan sangat puas dengan performanya. Layarnya jernih, kameranya bagus, dan baterai tahan lama. Namun, harganya cukup mahal dibandingkan dengan fitur yang ditawarkan. Meskipun begitu, secara keseluruhan saya senang dengan pembelian ini dan akan merekomendasikannya kepada teman-teman yang mencari ponsel baru.
    """,
    
    "Artikel Ilmiah": """
    Perubahan iklim telah menjadi masalah global yang semakin serius dalam beberapa dekade terakhir. Para ilmuwan telah meneliti dampak perubahan iklim terhadap keanekaragaman hayati di Indonesia. Hasil penelitian menunjukkan bahwa kenaikan suhu rata-rata sebesar 1,5 derajat Celsius dapat mengakibatkan kepunahan berbagai spesies endemik. Diperlukan tindakan segera untuk mengurangi emisi gas rumah kaca dan melindungi ekosistem yang terancam.
    """,
    
    "Cerita Pendek": """
    Desa kecil di kaki gunung itu selalu tenang. Setiap pagi, Pak Tono berjalan menyusuri sawah dengan cangkul di pundaknya. Dia tersenyum melihat padi yang mulai menguning. Tahun ini panennya akan berlimpah. Tiba-tiba, awan gelap menutupi matahari. Hujan akan segera turun, pikirnya. Dia bergegas pulang, tapi sebelum sampai di rumah bambu kecilnya, hujan deras sudah membasahi seluruh tubuhnya. Pak Tono tertawa. Dia selalu menikmati kejutan-kejutan kecil dalam hidupnya.
    """
}

# Text input options
input_option = st.radio(
    "Choose input method:",
    ["Enter text", "Upload file", "Use sample text"],
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
elif input_option == "Upload file":
    uploaded_file = st.file_uploader("Upload text file", type=['txt'])
    if uploaded_file is not None:
        user_input = uploaded_file.read().decode("utf-8")
        st.session_state.raw_text = user_input
        st.text_area("File content:", value=user_input, height=200, disabled=True)
    else:
        user_input = st.session_state.raw_text
else:  # Use sample text
    # Check if we need to fetch dynamic samples
    if not st.session_state.dynamic_samples or st.session_state.refresh_samples:
        with st.spinner("Fetching sample texts from Indonesian websites..."):
            try:
                # Fetch dynamic samples from real Indonesian websites
                dynamic_samples = get_dynamic_samples(2)  # Try to get 2 dynamic samples
                if dynamic_samples:
                    st.session_state.dynamic_samples = dynamic_samples
                st.session_state.refresh_samples = False
            except Exception as e:
                logger.error(f"Error fetching dynamic samples: {str(e)}")
                # Keep using static samples if dynamic ones fail
                if not st.session_state.dynamic_samples:
                    st.session_state.dynamic_samples = {}
    
    # Combine static and dynamic samples
    all_samples = {**sample_texts, **st.session_state.dynamic_samples}
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        sample_selection = st.selectbox(
            "Choose a sample Indonesian text:",
            options=list(all_samples.keys())
        )
    
    with col2:
        if st.button("🔄 Refresh Samples", key="refresh_samples"):
            st.session_state.refresh_samples = True
            st.rerun()
    
    # Create a stylized card for the sample
    st.markdown("""
    <div class="stCard">
        <h4 style="margin-top: 0;">Sample Text</h4>
    """, unsafe_allow_html=True)
    
    user_input = all_samples[sample_selection]
    st.text_area("", value=user_input, height=200, disabled=True, 
                label_visibility="collapsed")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("📋 Use this sample", key="use_sample"):
            st.session_state.raw_text = user_input
    
    # Close the card div
    st.markdown("</div>", unsafe_allow_html=True)

# Analysis options
st.header("🔍 Analysis Options")
col1, col2, col3, col4 = st.columns(4)

with col1:
    do_preprocessing = st.checkbox("Text Preprocessing", value=True)
    if st.button("Run Preprocessing", key="btn_preprocess", use_container_width=True):
        if not user_input:
            st.error("Please enter some text to analyze.")
        elif not st.session_state.api_key:
            st.warning("Please enter your GEMINI API key in the sidebar.")
        else:
            with st.spinner("Preprocessing text..."):
                # Store raw text
                st.session_state.raw_text = user_input
                
                st.info("Preprocessing text...")
                processed_text = preprocess_text(user_input)
                st.session_state.processed_text = processed_text
                st.success("Text preprocessing completed!")

with col2:
    do_sentiment = st.checkbox("Sentiment Analysis", value=True)
    if st.button("Run Sentiment", key="btn_sentiment", use_container_width=True):
        if not user_input:
            st.error("Please enter some text to analyze.")
        elif not st.session_state.api_key:
            st.warning("Please enter your GEMINI API key in the sidebar.")
        else:
            with st.spinner("Analyzing sentiment..."):
                # Store raw text
                st.session_state.raw_text = user_input
                
                # Ensure we have processed text
                if not st.session_state.processed_text:
                    processed_text = preprocess_text(user_input)
                    st.session_state.processed_text = processed_text
                
                st.info("Analyzing sentiment...")
                sentiment_result = analyze_sentiment(user_input, st.session_state.api_key)
                st.session_state.sentiment_result = sentiment_result
                st.success("Sentiment analysis completed!")

with col3:
    do_topic = st.checkbox("Topic Modeling", value=True)
    if st.button("Run Topics", key="btn_topics", use_container_width=True):
        if not user_input:
            st.error("Please enter some text to analyze.")
        elif not st.session_state.api_key:
            st.warning("Please enter your GEMINI API key in the sidebar.")
        else:
            with st.spinner("Extracting topics..."):
                # Store raw text
                st.session_state.raw_text = user_input
                
                # Ensure we have processed text
                if not st.session_state.processed_text:
                    processed_text = preprocess_text(user_input)
                    st.session_state.processed_text = processed_text
                else:
                    processed_text = st.session_state.processed_text
                
                st.info("Extracting topics...")
                topic_result = perform_topic_modeling(processed_text)
                st.session_state.topic_result = topic_result
                st.success("Topic modeling completed!")

with col4:
    st.checkbox("Comprehensive Analysis", value=True, key="do_comprehensive")
    if st.button("Full Analysis", key="btn_full", use_container_width=True, type="primary"):
        if not user_input:
            st.error("Please enter some text to analyze.")
        elif not st.session_state.api_key:
            st.warning("Please enter your GEMINI API key in the sidebar.")
        else:
            with st.spinner("Running complete analysis..."):
                # Store raw text
                st.session_state.raw_text = user_input
                
                progress_bar = st.progress(0)
                
                # Run text preprocessing
                st.info("Preprocessing text...")
                processed_text = preprocess_text(user_input)
                st.session_state.processed_text = processed_text
                progress_bar.progress(25)
                
                # Run sentiment analysis
                st.info("Analyzing sentiment...")
                sentiment_result = analyze_sentiment(user_input, st.session_state.api_key)
                st.session_state.sentiment_result = sentiment_result
                progress_bar.progress(50)
                
                # Run topic modeling
                st.info("Extracting topics...")
                topic_result = perform_topic_modeling(processed_text)
                st.session_state.topic_result = topic_result
                progress_bar.progress(75)
                
                # Run comprehensive analysis
                st.info("Getting comprehensive insights from GEMINI...")
                gemini_analysis = analyze_text_with_gemini(user_input, st.session_state.api_key)
                st.session_state.full_analysis = gemini_analysis
                progress_bar.progress(100)
                
                st.success("Full analysis completed!")
                time.sleep(0.5)  # Brief pause to show completion

if not st.session_state.api_key:
    st.warning("Please enter your GEMINI API key in the sidebar to perform the analysis.")

# Display results if available
if st.session_state.raw_text:
    st.header("📊 Analysis Results")
    
    # Save analysis results to database when all analysis is completed
    if (st.session_state.processed_text and 
        st.session_state.sentiment_result and 
        st.session_state.topic_result and 
        st.session_state.full_analysis):
        
        # Create a "Save Results" button
        if st.button("💾 Save Analysis Results to Database", key="btn_save"):
            with st.spinner("Saving results to database..."):
                # Get word count
                word_count = count_words(st.session_state.raw_text)
                
                # Get text type (if selected from sample)
                text_type = None
                if input_option == "Use sample text":
                    # Determine text type from all samples (static and dynamic)
                    all_samples = {**sample_texts, **st.session_state.dynamic_samples}
                    for sample_name, sample_content in all_samples.items():
                        if user_input.strip() == sample_content.strip():
                            text_type = sample_name
                            break
                
                # Convert processed text to string
                preprocessed_text_str = " ".join(st.session_state.processed_text)
                
                # Save to database
                record_id = save_analysis(
                    text=st.session_state.raw_text,
                    text_type=text_type,
                    word_count=word_count,
                    preprocessed_text=preprocessed_text_str,
                    sentiment_analysis=st.session_state.sentiment_result,
                    topic_modeling=st.session_state.topic_result
                )
                
                if record_id:
                    st.success(f"Analysis results saved to database with ID: {record_id}")
                else:
                    st.error("Failed to save analysis results to database.")
    
    tabs = st.tabs(["Overview", "Preprocessing", "Sentiment Analysis", "Topic Modeling", "Full Analysis", "History", "Batch Processing"])
    
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
            
            # Add emotional radar chart
            st.subheader("Emotional Analysis")
            
            # Create emotion data (based on sentiment scores)
            emotions = {
                "Senang (Happy)": result.get("positive_score", 0) * 0.8,
                "Kagum (Amazed)": result.get("positive_score", 0) * 0.6,
                "Tertarik (Interested)": result.get("positive_score", 0) * 0.7,
                "Tenang (Calm)": result.get("neutral_score", 0) * 0.8,
                "Bingung (Confused)": result.get("neutral_score", 0) * 0.6,
                "Marah (Angry)": result.get("negative_score", 0) * 0.7,
                "Sedih (Sad)": result.get("negative_score", 0) * 0.8,
                "Kecewa (Disappointed)": result.get("negative_score", 0) * 0.6,
            }
            
            # Create radar chart
            categories = list(emotions.keys())
            values = list(emotions.values())
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Emotional Analysis',
                line_color='purple',
                fillcolor='rgba(128, 0, 128, 0.3)'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=False,
                title="Emotional Dimensions in Text",
                height=450
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Sentiment trend visualization (simulate with random data for demonstration)
            if len(st.session_state.raw_text.split('.')) > 5:
                st.subheader("Sentiment Flow")
                st.markdown("Sentiment analysis across text segments")
                
                # Get sentences and simulate sentiment for each
                import random
                sentences = [s.strip() for s in st.session_state.raw_text.split('.') if s.strip()]
                sentences = sentences[:10]  # Take first 10 sentences max
                
                # Get main sentiment as baseline
                main_sentiment = result.get("sentiment", "Neutral")
                baseline = {
                    "Positive": 0.7,
                    "Neutral": 0.5,
                    "Negative": 0.3
                }.get(main_sentiment, 0.5)
                
                # Generate sentiment flow data
                sentiment_flow = []
                for i, _ in enumerate(sentences):
                    variation = random.uniform(-0.2, 0.2)
                    sent_value = min(1.0, max(0.0, baseline + variation))
                    sentiment_flow.append(sent_value)
                
                # Create a dataframe for the chart
                flow_df = pd.DataFrame({
                    'Segment': [f"Segment {i+1}" for i in range(len(sentences))],
                    'Sentiment': sentiment_flow
                })
                
                # Create line chart
                fig = px.line(
                    flow_df, 
                    x='Segment', 
                    y='Sentiment',
                    markers=True,
                    title="Sentiment Flow Across Text Segments",
                    color_discrete_sequence=['purple']
                )
                
                # Add reference lines
                fig.add_shape(type="line", line_dash="dash", x0=0, y0=0.7, x1=len(sentences)-1, y1=0.7,
                              line=dict(color="green", width=1))
                fig.add_shape(type="line", line_dash="dash", x0=0, y0=0.5, x1=len(sentences)-1, y1=0.5,
                              line=dict(color="gray", width=1))
                fig.add_shape(type="line", line_dash="dash", x0=0, y0=0.3, x1=len(sentences)-1, y1=0.3,
                              line=dict(color="red", width=1))
                
                # Add annotations
                fig.add_annotation(x=len(sentences)-1, y=0.7, text="Positive", showarrow=False, yshift=10, 
                                  font=dict(color="green"))
                fig.add_annotation(x=len(sentences)-1, y=0.5, text="Neutral", showarrow=False, yshift=10, 
                                  font=dict(color="gray"))
                fig.add_annotation(x=len(sentences)-1, y=0.3, text="Negative", showarrow=False, yshift=10, 
                                  font=dict(color="red"))
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
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
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Pie chart
                    fig = px.pie(
                        topic_df, 
                        values='percentage', 
                        names='topic', 
                        title='Topic Distribution in Text',
                        color_discrete_sequence=px.colors.qualitative.Plotly,
                        hole=0.4
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Bar chart alternative view
                    fig = px.bar(
                        topic_df,
                        x='topic',
                        y='percentage',
                        title='Topic Prominence',
                        color='percentage',
                        color_continuous_scale='Viridis',
                        text_auto='.0%'
                    )
                    fig.update_layout(yaxis_tickformat='.0%')
                    st.plotly_chart(fig, use_container_width=True)
            
            # Topic network visualization
            st.subheader("Topic Network")
            st.markdown("Network visualization of topics and key terms connections")
            
            # Create network visualization from topics
            if len(topics) > 1:
                # Prepare data for network graph
                import networkx as nx
                
                # Create a graph
                G = nx.Graph()
                
                # Add nodes for topics
                for i, topic in enumerate(topics):
                    G.add_node(f"Topic {i+1}", type="topic", size=20)
                    
                    # Get top words from each topic
                    top_words = [word for word, _ in topic[:5]]
                    
                    # Add nodes for words and edges to topic
                    for word, weight in topic[:5]:
                        if word not in G:
                            G.add_node(word, type="word", size=10)
                        G.add_edge(f"Topic {i+1}", word, weight=weight)
                
                # Connect words that appear in multiple topics
                for i, topic1 in enumerate(topics):
                    words1 = set([word for word, _ in topic1])
                    for j, topic2 in enumerate(topics):
                        if i >= j:  # Avoid duplicate connections
                            continue
                        words2 = set([word for word, _ in topic2])
                        common_words = words1.intersection(words2)
                        
                        # If topics share words, add edge between them
                        if common_words:
                            G.add_edge(f"Topic {i+1}", f"Topic {j+1}", 
                                      weight=len(common_words) / 5)  # Normalize weight
                
                # Create positions for nodes
                pos = nx.spring_layout(G, seed=42)
                
                # Prepare data for plotly
                edge_x = []
                edge_y = []
                for edge in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                
                # Create edges trace
                edge_trace = go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=0.8, color='#888'),
                    hoverinfo='none',
                    mode='lines')
                
                # Create nodes trace
                node_x = []
                node_y = []
                node_text = []
                node_size = []
                node_color = []
                
                for node in G.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    node_text.append(node)
                    
                    # Set size based on node type
                    if G.nodes[node]['type'] == 'topic':
                        node_size.append(25)
                        node_color.append(0)  # Topics use first color
                    else:
                        node_size.append(15)
                        node_color.append(1)  # Words use second color
                
                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers+text',
                    hoverinfo='text',
                    text=node_text,
                    textposition="top center",
                    marker=dict(
                        showscale=False,
                        colorscale='Viridis',
                        color=node_color,
                        size=node_size,
                        line=dict(width=2)
                    )
                )
                
                # Create figure
                fig = go.Figure(data=[edge_trace, node_trace],
                             layout=go.Layout(
                                showlegend=False,
                                hovermode='closest',
                                margin=dict(b=20,l=5,r=5,t=40),
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                title="Topic and Word Relationships",
                                height=600
                              )
                          )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Multiple topics are needed to generate a network visualization.")
                
            # Topic heatmap
            if len(topics) > 1:
                st.subheader("Topic-Word Heatmap")
                
                # Create a heatmap of topic-word associations
                heatmap_data = []
                words = set()
                
                # Collect all unique words from topics
                for topic in topics:
                    words.update([word for word, _ in topic[:5]])
                
                words = list(words)
                
                # Create heatmap data
                for i, topic in enumerate(topics):
                    topic_dict = dict(topic)
                    row = [topic_dict.get(word, 0) for word in words]
                    heatmap_data.append(row)
                
                # Create heatmap
                fig = go.Figure(data=go.Heatmap(
                    z=heatmap_data,
                    x=words,
                    y=[f"Topic {i+1}" for i in range(len(topics))],
                    colorscale='Viridis',
                    hoverongaps=False))
                
                fig.update_layout(
                    title="Word Importance Across Topics",
                    height=400,
                    xaxis_title="Words",
                    yaxis_title="Topics"
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
    
    with tabs[5]:
        st.subheader("Analysis History")
        
        # Get history from database
        history = get_analysis_history(limit=20)
        
        if history:
            st.markdown("### Recent Analysis Records")
            
            # Create a dataframe for display
            history_df = pd.DataFrame(history)
            
            # Format timestamp for better display
            if 'timestamp' in history_df.columns:
                history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
                history_df['timestamp'] = history_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            
            # Add a view button column
            history_df['action'] = 'View'
            
            # Display as table
            selected_indices = st.dataframe(
                history_df[['id', 'timestamp', 'text_type', 'word_count', 'has_sentiment', 'has_topics', 'text', 'action']], 
                column_config={
                    "id": st.column_config.NumberColumn("ID", width="small"),
                    "timestamp": st.column_config.DatetimeColumn("Date & Time", width="medium"),
                    "text_type": st.column_config.TextColumn("Type", width="small"),
                    "word_count": st.column_config.NumberColumn("Words", width="small"),
                    "has_sentiment": st.column_config.CheckboxColumn("Sentiment", width="small"),
                    "has_topics": st.column_config.CheckboxColumn("Topics", width="small"),
                    "text": st.column_config.TextColumn("Text Preview", width="large"),
                    "action": st.column_config.ButtonColumn("Action", width="small")
                },
                height=400,
                use_container_width=True,
                hide_index=True
            )
            
    with tabs[6]:
        st.subheader("📚 Batch Processing")
        
        # Initialize session state for batch processing
        if 'batch_texts' not in st.session_state:
            st.session_state.batch_texts = []
            st.session_state.batch_text_types = []
            st.session_state.batch_results = None
            st.session_state.batch_report = None
            st.session_state.batch_comparison = None
        
        st.markdown("""
        <div class="stCard">
            <h3 style="margin-top: 0;">Analyze Multiple Documents</h3>
            <p>Process multiple Indonesian text documents simultaneously and compare their results.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Batch options
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Add Documents to Batch")
            
            # Text input options
            input_method = st.radio(
                "Add document using:",
                ["Enter text", "Upload file", "Use sample"],
                horizontal=True,
                key="batch_input_method"
            )
            
            # Add document form
            with st.form(key="add_document_form"):
                if input_method == "Enter text":
                    batch_text = st.text_area(
                        "Enter Indonesian text:",
                        height=150,
                        placeholder="Paste your Indonesian text here..."
                    )
                elif input_method == "Upload file":
                    uploaded_file = st.file_uploader("Upload text file", type=['txt'], key="batch_file_uploader")
                    batch_text = uploaded_file.read().decode("utf-8") if uploaded_file else ""
                else:  # Use sample
                    # Combine static and dynamic samples
                    all_samples = {**sample_texts, **st.session_state.dynamic_samples}
                    sample_selection = st.selectbox(
                        "Choose a sample Indonesian text:",
                        options=list(all_samples.keys()),
                        key="batch_sample_selection"
                    )
                    batch_text = all_samples[sample_selection]
                    st.text_area("Preview:", value=batch_text[:200] + "...", height=100, disabled=True)
                
                text_type = st.selectbox(
                    "Document type:",
                    options=["News", "Review", "Academic", "Social Media", "Story", "Other"],
                    key="batch_text_type"
                )
                
                submit_button = st.form_submit_button("Add to Batch", type="primary")
                
                if submit_button and batch_text:
                    st.session_state.batch_texts.append(batch_text)
                    st.session_state.batch_text_types.append(text_type)
                    st.success(f"Document added! Batch now contains {len(st.session_state.batch_texts)} documents.")
        
        with col2:
            st.markdown("### Batch Queue")
            
            # Display current batch
            if st.session_state.batch_texts:
                st.markdown(f"**{len(st.session_state.batch_texts)} documents in queue:**")
                for i, (text, text_type) in enumerate(zip(st.session_state.batch_texts, st.session_state.batch_text_types)):
                    preview = text[:50] + "..." if len(text) > 50 else text
                    st.markdown(f"{i+1}. **{text_type}**: {preview}")
                
                if st.button("Clear All Documents", key="clear_batch"):
                    st.session_state.batch_texts = []
                    st.session_state.batch_text_types = []
                    st.session_state.batch_results = None
                    st.session_state.batch_report = None
                    st.session_state.batch_comparison = None
                    st.rerun()
            else:
                st.info("No documents in batch queue yet. Add documents using the form.")
        
        # Batch processing options
        if st.session_state.batch_texts:
            st.markdown("### Process Batch")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                do_batch_preprocess = st.checkbox("Text Preprocessing", value=True, key="batch_preprocess")
            with col2:
                do_batch_sentiment = st.checkbox("Sentiment Analysis", value=True, key="batch_sentiment")
            with col3:
                do_batch_topic = st.checkbox("Topic Modeling", value=True, key="batch_topic")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                save_to_db = st.checkbox("Save results to database", value=True, key="batch_save_db")
            
            with col2:
                max_workers = st.slider("Parallel Workers", min_value=1, max_value=5, value=3, key="batch_workers")
            
            process_config = {
                'do_preprocessing': do_batch_preprocess,
                'do_sentiment': do_batch_sentiment,
                'do_topic': do_batch_topic
            }
            
            if st.button("🚀 Process Batch", type="primary", key="run_batch", use_container_width=True):
                if not st.session_state.api_key:
                    st.warning("Please enter your GEMINI API key in the sidebar.")
                else:
                    with st.spinner("Processing batch of documents..."):
                        try:
                            # Run batch processing
                            st.session_state.batch_results = batch_process(
                                st.session_state.batch_texts,
                                st.session_state.batch_text_types,
                                st.session_state.api_key,
                                process_config,
                                save_to_db,
                                max_workers
                            )
                            
                            # Generate report
                            st.session_state.batch_report = generate_batch_report(st.session_state.batch_results)
                            
                            # Generate comparison if more than 1 document
                            if len(st.session_state.batch_texts) > 1:
                                st.session_state.batch_comparison = compare_texts(st.session_state.batch_results)
                            
                            st.success(f"Successfully processed {len(st.session_state.batch_results)} documents!")
                        except Exception as e:
                            st.error(f"Error processing batch: {str(e)}")
            
            # Display batch results if available
            if st.session_state.batch_results:
                st.markdown("---")
                st.markdown("## Batch Results")
                
                # Create tabs for results, comparison, and report
                batch_result_tabs = st.tabs(["Documents", "Comparison", "Summary Report"])
                
                with batch_result_tabs[0]:
                    st.markdown("### Individual Document Results")
                    
                    # Create expandable sections for each document
                    for i, result in enumerate(st.session_state.batch_results):
                        with st.expander(f"Document {i+1}: {result.get('text_type', 'Unknown')}"):
                            if result.get('status') == 'failed':
                                st.error(f"Processing failed: {', '.join(result.get('errors', ['Unknown error']))}")
                                continue
                            
                            # Document info
                            st.markdown(f"**Document Type:** {result.get('text_type', 'Unknown')}")
                            st.markdown(f"**Word Count:** {result.get('word_count', 0)}")
                            
                            # Preview
                            with st.expander("Text Preview"):
                                st.text_area("", value=result.get('text', '')[:500] + "..." if len(result.get('text', '')) > 500 else result.get('text', ''), 
                                            height=100, disabled=True)
                            
                            # Display results
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                if result.get('sentiment_result'):
                                    sentiment = result['sentiment_result'].get('sentiment', 'Unknown')
                                    pos_score = result['sentiment_result'].get('positive_score', 0) * 100
                                    neu_score = result['sentiment_result'].get('neutral_score', 0) * 100
                                    neg_score = result['sentiment_result'].get('negative_score', 0) * 100
                                    
                                    sentiment_color = {
                                        "Positive": "green",
                                        "Negative": "red",
                                        "Neutral": "gray"
                                    }.get(sentiment, "blue")
                                    
                                    st.markdown(f"**Sentiment:** <span style='color:{sentiment_color}'>{sentiment}</span>", unsafe_allow_html=True)
                                    
                                    # Small bar chart for sentiment scores
                                    sentiment_data = {
                                        'Sentiment': ['Positive', 'Neutral', 'Negative'],
                                        'Score': [pos_score, neu_score, neg_score]
                                    }
                                    sentiment_df = pd.DataFrame(sentiment_data)
                                    fig = px.bar(
                                        sentiment_df, 
                                        x='Score', 
                                        y='Sentiment', 
                                        orientation='h',
                                        color='Sentiment',
                                        color_discrete_map={
                                            'Positive': 'green',
                                            'Neutral': 'gray',
                                            'Negative': 'red'
                                        },
                                        text_auto='.0f'
                                    )
                                    fig.update_layout(height=200, xaxis_title="", yaxis_title="", xaxis_range=[0, 100])
                                    st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                if result.get('topic_result'):
                                    st.markdown("**Top Topics:**")
                                    
                                    # Get all words from topics and their weights
                                    all_words = []
                                    for topic in result['topic_result']:
                                        for word, weight in topic[:5]:  # Get top 5 words from each topic
                                            all_words.append((word, weight))
                                    
                                    # Create word count dataframe
                                    words_df = pd.DataFrame(all_words, columns=['Word', 'Weight'])
                                    words_df = words_df.groupby('Word').sum().reset_index().sort_values('Weight', ascending=False).head(10)
                                    
                                    # Plot horizontal bar chart
                                    fig = px.bar(
                                        words_df, 
                                        x='Weight', 
                                        y='Word', 
                                        orientation='h',
                                        color='Weight',
                                        color_continuous_scale='viridis',
                                        text_auto='.2f'
                                    )
                                    fig.update_layout(height=300, yaxis=dict(autorange="reversed"))
                                    st.plotly_chart(fig, use_container_width=True)
                
                with batch_result_tabs[1]:
                    if st.session_state.batch_comparison and 'error' not in st.session_state.batch_comparison:
                        st.markdown("### Document Comparison")
                        
                        # Sentiment comparison
                        st.subheader("Sentiment Comparison")
                        if st.session_state.batch_comparison.get('sentiment_comparison'):
                            sentiment_comp = pd.DataFrame(st.session_state.batch_comparison['sentiment_comparison'])
                            
                            # Radar chart for sentiment comparison
                            fig = go.Figure()
                            
                            for i, row in sentiment_comp.iterrows():
                                fig.add_trace(go.Scatterpolar(
                                    r=[row['positive_score']*100, row['neutral_score']*100, row['negative_score']*100],
                                    theta=['Positive', 'Neutral', 'Negative'],
                                    fill='toself',
                                    name=f"Doc {row['text_num']}: {row['text_type']}"
                                ))
                            
                            fig.update_layout(
                                polar=dict(
                                    radialaxis=dict(
                                        visible=True,
                                        range=[0, 100]
                                    )),
                                showlegend=True,
                                title="Sentiment Comparison"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Length comparison
                        st.subheader("Length Comparison")
                        if st.session_state.batch_comparison.get('length_comparison'):
                            length_comp = pd.DataFrame(st.session_state.batch_comparison['length_comparison'])
                            
                            # Create a bar chart
                            fig = px.bar(
                                length_comp,
                                x='text_num',
                                y='word_count',
                                color='text_type',
                                labels={'text_num': 'Document', 'word_count': 'Word Count'},
                                title="Word Count Comparison",
                                text_auto='.0f'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Topic similarity
                        st.subheader("Topic Similarity")
                        if st.session_state.batch_comparison.get('topic_similarity'):
                            topic_sim = pd.DataFrame(st.session_state.batch_comparison['topic_similarity'])
                            
                            # Create a heatmap-like visualization
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                fig = px.bar(
                                    topic_sim,
                                    x='text_pair',
                                    y='similarity_score',
                                    color='similarity_score',
                                    color_continuous_scale='viridis',
                                    title="Topic Similarity Between Documents",
                                    text_auto='.2f'
                                )
                                fig.update_layout(yaxis_range=[0, 1])
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                # Display common words
                                st.markdown("### Common Words")
                                for i, row in topic_sim.iterrows():
                                    st.markdown(f"**{row['text_pair']}:**")
                                    if row['common_words']:
                                        st.markdown(", ".join(row['common_words'][:8]))
                                    else:
                                        st.markdown("No common words")
                                    st.markdown("---")
                    else:
                        st.info("Comparison requires at least two successfully processed documents.")
                
                with batch_result_tabs[2]:
                    if st.session_state.batch_report:
                        st.markdown("### Batch Processing Report")
                        
                        # Overview metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Documents", st.session_state.batch_report.get('total_texts', 0))
                        with col2:
                            st.metric("Successful", st.session_state.batch_report.get('successful', 0))
                        with col3:
                            st.metric("Failed", st.session_state.batch_report.get('failed', 0))
                        with col4:
                            avg_words = st.session_state.batch_report.get('word_count_stats', {}).get('avg', 0)
                            st.metric("Avg Words", f"{int(avg_words)}")
                        
                        # Sentiment distribution
                        if st.session_state.batch_report.get('sentiment_distribution'):
                            st.subheader("Sentiment Distribution")
                            sentiment_counts = st.session_state.batch_report['sentiment_distribution']
                            sentiment_df = pd.DataFrame({
                                'Sentiment': list(sentiment_counts.keys()),
                                'Count': list(sentiment_counts.values())
                            })
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Pie chart
                                fig = px.pie(
                                    sentiment_df,
                                    values='Count',
                                    names='Sentiment',
                                    title='Sentiment Distribution',
                                    color='Sentiment',
                                    color_discrete_map={
                                        'Positive': 'green',
                                        'Neutral': 'gray',
                                        'Negative': 'red',
                                        'Unknown': 'blue'
                                    }
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                # Bar chart
                                fig = px.bar(
                                    sentiment_df,
                                    x='Sentiment',
                                    y='Count',
                                    title='Sentiment Count',
                                    color='Sentiment',
                                    color_discrete_map={
                                        'Positive': 'green',
                                        'Neutral': 'gray',
                                        'Negative': 'red',
                                        'Unknown': 'blue'
                                    },
                                    text_auto='.0f'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        
                        # Word count statistics
                        if st.session_state.batch_report.get('word_count_stats'):
                            st.subheader("Word Count Statistics")
                            
                            word_stats = st.session_state.batch_report['word_count_stats']
                            stats_df = pd.DataFrame({
                                'Statistic': ['Minimum', 'Maximum', 'Average', 'Total'],
                                'Count': [
                                    word_stats.get('min', 0),
                                    word_stats.get('max', 0),
                                    round(word_stats.get('avg', 0), 1),
                                    word_stats.get('total', 0)
                                ]
                            })
                            
                            fig = px.bar(
                                stats_df,
                                x='Statistic',
                                y='Count',
                                title='Word Count Statistics',
                                color='Statistic',
                                text_auto='.0f'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Topic summary
                        if st.session_state.batch_report.get('topic_summary'):
                            st.subheader("Common Topics Across Documents")
                            
                            topic_summary = st.session_state.batch_report['topic_summary']
                            topic_df = pd.DataFrame(topic_summary)
                            
                            # Bar chart of most common words
                            fig = px.bar(
                                topic_df.head(15),
                                x='count',
                                y='word',
                                orientation='h',
                                title='Most Common Words Across All Documents',
                                color='count',
                                color_continuous_scale='viridis',
                                text_auto='.0f'
                            )
                            fig.update_layout(yaxis=dict(autorange="reversed"))
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Run batch processing to generate a summary report.")
            
            # Handle viewing a record
            if selected_indices:
                for selected_index in selected_indices.rows:
                    record_id = history_df.iloc[selected_index].id
                    
                    # Get full record details
                    record = get_analysis_by_id(record_id)
                    
                    if record:
                        st.markdown(f"### Analysis Record #{record_id}")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### Text Information")
                            st.markdown(f"**Type:** {record['text_type'] or 'Not specified'}")
                            st.markdown(f"**Date:** {record['timestamp']}")
                            st.markdown(f"**Word Count:** {record['word_count']}")
                            
                            # Text preview with expandable view
                            with st.expander("View Full Text", expanded=False):
                                st.text_area("", value=record['text'], height=200, disabled=True)
                        
                        with col2:
                            # Display sentiment result if available
                            if record['sentiment_analysis']:
                                sentiment_data = record['sentiment_analysis']
                                
                                sentiment = sentiment_data.get("sentiment", "Unknown")
                                sentiment_color = {
                                    "Positive": "green",
                                    "Negative": "red",
                                    "Neutral": "gray"
                                }.get(sentiment, "blue")
                                
                                st.markdown(f"**Sentiment:** <span style='color:{sentiment_color}'>{sentiment}</span>", unsafe_allow_html=True)
                                
                                # Show sentiment scores
                                scores = {
                                    "Positive": sentiment_data.get("positive_score", 0),
                                    "Neutral": sentiment_data.get("neutral_score", 0),
                                    "Negative": sentiment_data.get("negative_score", 0)
                                }
                                
                                # Create a small horizontal bar chart for sentiment scores
                                score_df = pd.DataFrame({
                                    'Sentiment': list(scores.keys()),
                                    'Score': list(scores.values())
                                })
                                
                                fig = px.bar(
                                    score_df,
                                    x='Score',
                                    y='Sentiment',
                                    orientation='h',
                                    title="Sentiment Scores",
                                    color='Sentiment',
                                    color_discrete_map={
                                        'Positive': 'green',
                                        'Neutral': 'gray',
                                        'Negative': 'red'
                                    }
                                )
                                fig.update_layout(height=200)
                                st.plotly_chart(fig, use_container_width=True)
                        
                        # Display topics if available
                        if record['topic_modeling']:
                            st.markdown("#### Top Topics")
                            topic_data = record['topic_modeling']
                            
                            # Display first two topics as examples
                            for i, topic in enumerate(topic_data[:2]):
                                with st.expander(f"Topic {i+1}", expanded=False):
                                    # Display words and weights
                                    topic_df = pd.DataFrame(topic, columns=["Word", "Weight"])
                                    st.dataframe(topic_df, use_container_width=True, hide_index=True)
        else:
            st.info("No analysis history found. Complete and save an analysis to see it here.")

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
