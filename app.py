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
import time
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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
    sample_selection = st.selectbox(
        "Choose a sample Indonesian text:",
        options=list(sample_texts.keys())
    )
    
    user_input = sample_texts[sample_selection]
    if st.button("Use this sample", key="use_sample"):
        st.session_state.raw_text = user_input
    
    st.text_area("Sample text:", value=user_input, height=200, disabled=True)

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
                    # Determine text type from the sample text
                    for sample_name, sample_content in sample_texts.items():
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
    
    tabs = st.tabs(["Overview", "Preprocessing", "Sentiment Analysis", "Topic Modeling", "Full Analysis", "History"])
    
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
