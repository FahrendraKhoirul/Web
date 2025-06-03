import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import torch

#  ====================== Utility Functions ======================

def simulate_topic_prediction(text):
    """Topic model prediction - replace with your actual model"""
    topics = ['product', 'customer_service', 'shipping']
    predictions = {}
    
    token_result = pipeline.tokenizer(text, return_tensors="pt")
    input_ids = token_result['input_ids']
    attention_mask = token_result['attention_mask']
    
    with torch.no_grad():
        outputs = pipeline.aspect_model(input_ids=input_ids, attention_mask=attention_mask)

    # st.write("Topic Model output:", outputs)
    
    # Convert model outputs to probabilities
    # The output is already sigmoid result, use directly
    if hasattr(outputs, 'logits'):
        probs = outputs.logits.squeeze().numpy()
    else:
        probs = outputs.squeeze().numpy()
    
    # Create predictions dictionary
    for i, topic in enumerate(topics):
        predictions[topic] = float(probs[i]) if len(probs.shape) > 0 else float(probs)
    
    return predictions

def simulate_sentiment_prediction(text):
    """Sentiment prediction - replace with your actual model"""
    sentiments = ['positive', 'neutral', 'negative']
    
    token_result = pipeline.tokenizer(text, return_tensors="pt")
    input_ids = token_result['input_ids']
    attention_mask = token_result['attention_mask']
    
    with torch.no_grad():
        outputs = pipeline.sentiment_model(input_ids=input_ids, attention_mask=attention_mask)
    
    # st.write("Sentiment Model output:", outputs)
    
    # Convert model outputs to probabilities
    # The output is already softmax result, use directly
    if hasattr(outputs, 'logits'):
        probs = outputs.logits.squeeze().numpy()
    else:
        probs = outputs.squeeze().numpy()
    
    # st.write("Sentiment Model probabilities:", probs)
    # Get the predicted sentiment
    predicted_idx = np.argmax(probs)
    predicted_sentiment = sentiments[predicted_idx]
    confidence = float(probs[predicted_idx])
    
    return {
        'sentiment': predicted_sentiment,
        'confidence': confidence,
        'all_probs': {sentiments[i]: float(probs[i]) for i in range(len(sentiments))}
    }

def display_predictions(text, topic_predictions, sentiment_prediction):
    """Display prediction results"""
    st.markdown("---")
    st.subheader("🎯 Classification Results")
    
    # Display input text
    st.markdown("**Input Text:**")
    st.info(text)
    
    col_topic, col_sentiment = st.columns(2)
    
    with col_topic:
        st.markdown("**🏷️ Topic Classification (Multi-label):**")
        
        for topic, prob in topic_predictions.items():
            if prob >= 0.5:  # Fixed threshold
                confidence_class = "topic-positive" if prob > 0.7 else "topic-neutral"
                emoji = "✅" if prob > 0.7 else "⚠️"
                
                result_html = f"""
                <div class="prediction-result {confidence_class}">
                    {emoji} <strong>{topic.replace('_', ' ').title()}</strong>
                    <br>Confidence: {prob:.2%}
                </div>
                """
                st.markdown(result_html, unsafe_allow_html=True)
        
        # Show chart
        fig_topic = create_topic_chart(topic_predictions)
        st.plotly_chart(fig_topic, use_container_width=True)
    
    with col_sentiment:
        st.markdown("**😊 Sentiment Analysis:**")
        
        sentiment = sentiment_prediction['sentiment']
        confidence = sentiment_prediction['confidence']
        
        sentiment_emoji = {"positive": "😊", "neutral": "😐", "negative": "😞"}
        sentiment_class = f"topic-{sentiment}"
        
        result_html = f"""
        <div class="prediction-result {sentiment_class}">
            {sentiment_emoji[sentiment]} <strong>{sentiment.title()}</strong>
            <br>Confidence: {confidence:.2%}
        </div>
        """
        st.markdown(result_html, unsafe_allow_html=True)
        
        # Show chart
        fig_sentiment = create_sentiment_chart(sentiment_prediction)
        st.plotly_chart(fig_sentiment, use_container_width=True)
    
    # Store in session state for statistics
    if 'classification_history' not in st.session_state:
        st.session_state.classification_history = []
    
    st.session_state.classification_history.append({
        'text': text,
        'topics': topic_predictions,
        'sentiment': sentiment_prediction,
        'confidence': np.mean(list(topic_predictions.values()) + [sentiment_prediction['confidence']]),
        'timestamp': datetime.now()
    })

def create_topic_chart(predictions):
    """Create topic prediction chart"""
    topics = list(predictions.keys())
    probabilities = list(predictions.values())
    
    fig = go.Figure(data=[
        go.Bar(
            x=[t.replace('_', ' ').title() for t in topics],
            y=probabilities,
            marker_color=['#28a745' if p >= 0.5 else '#6c757d' for p in probabilities]
        )
    ])
    
    fig.update_layout(
        title="Topic Classification Probabilities",
        xaxis_title="Topics",
        yaxis_title="Probability",
        height=300,
        showlegend=False
    )
    
    fig.add_hline(y=0.5, line_dash="dash", line_color="red", 
                  annotation_text="Threshold (0.5)")
    
    return fig

def create_sentiment_chart(prediction):
    """Create sentiment prediction chart"""
    sentiments = ['positive', 'neutral', 'negative']
    
    # Use all probabilities if available, otherwise create from single prediction
    if 'all_probs' in prediction:
        probs = [prediction['all_probs'][s] for s in sentiments]
    else:
        # Fallback to original method
        current_sentiment = prediction['sentiment']
        confidence = prediction['confidence']
        probs = [0.1, 0.1, 0.1]
        idx = sentiments.index(current_sentiment)
        probs[idx] = confidence
        remaining = (1.0 - confidence) / 2
        for i, _ in enumerate(probs):
            if i != idx:
                probs[i] = remaining
    
    colors = ['#28a745', '#ffc107', '#dc3545']
    
    # Create donut chart for single prediction
    fig = go.Figure(data=[go.Pie(
        labels=[s.title() for s in sentiments],
        values=probs,
        hole=0.3,  # Creates donut effect
        marker_colors=colors,
        textinfo='label+percent',
        textposition='auto'
    )])
    
    fig.update_layout(
        title="Sentiment Analysis Probabilities",
        height=300,
        showlegend=False,
        margin=dict(t=50, b=20, l=20, r=20)
    )
    
    return fig

def process_batch_classification(df, text_column, max_results=10):
    """Process batch classification"""
    st.subheader("🔄 Batch Processing Results")
    
    progress_bar = st.progress(0)
    results = []
    all_topic_predictions = []
    all_sentiment_predictions = []
    
    for i, text in enumerate(df[text_column].values[:max_results]):
        if isinstance(text, str) and text.strip():
            try:
                topic_pred = simulate_topic_prediction(text)
                sentiment_pred = simulate_sentiment_prediction(text)
                
                # Store for visualization
                all_topic_predictions.append(topic_pred)
                all_sentiment_predictions.append(sentiment_pred)
                
                results.append({
                    'text': text[:100] + '...' if len(text) > 100 else text,
                    'topics': ', '.join([t for t, p in topic_pred.items() if p >= 0.5]),
                    'sentiment': sentiment_pred['sentiment'],
                    'sentiment_confidence': sentiment_pred['confidence']
                })
            except Exception as e:
                st.error(f"Error processing text {i+1}: {str(e)}")
                continue
        
        progress_bar.progress((i + 1) / min(len(df), max_results))
    
    # Display results table
    if results:
        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True)
        
        # Create visualization section
        st.markdown("---")
        st.subheader("📊 Batch Analysis Visualization")
        
        col_topic_viz, col_sentiment_viz = st.columns(2)
        
        with col_topic_viz:
            st.markdown("**Topic Distribution**")
            create_batch_topic_chart(all_topic_predictions)
        
        with col_sentiment_viz:
            st.markdown("**Sentiment Distribution**")
            create_batch_sentiment_chart(all_sentiment_predictions)
        
        # Download results
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results",
            data=csv,
            file_name=f"classification_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
def create_batch_topic_chart(all_predictions):
    """Create batch topic analysis chart"""
    topics = ['product', 'customer_service', 'shipping']
    topic_counts = {topic: 0 for topic in topics}
    total_texts = len(all_predictions)
    
    # Count how many texts were classified for each topic (above threshold)
    for pred in all_predictions:
        for topic, prob in pred.items():
            if prob >= 0.5:
                topic_counts[topic] += 1
    
    # Convert to percentages
    topic_percentages = {topic: (count / total_texts) * 100 for topic, count in topic_counts.items()}
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=[t.replace('_', ' ').title() for t in topics],
            y=list(topic_percentages.values()),
            marker_color=['#28a745', '#17a2b8', '#ffc107'],
            text=[f'{v:.1f}%' for v in topic_percentages.values()],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title=f"Topic Distribution Across {total_texts} Texts",
        xaxis_title="Topics",
        yaxis_title="Percentage of Texts (%)",
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_batch_sentiment_chart(all_predictions):
    """Create batch sentiment analysis chart (rounded/donut)"""
    sentiments = ['positive', 'neutral', 'negative']
    sentiment_counts = {sentiment: 0 for sentiment in sentiments}
    total_texts = len(all_predictions)
    
    # Count sentiment predictions
    for pred in all_predictions:
        sentiment = pred['sentiment']
        sentiment_counts[sentiment] += 1
    
    # Convert to percentages
    sentiment_percentages = [(count / total_texts) * 100 for count in sentiment_counts.values()]
    
    # Create donut chart
    colors = ['#28a745', '#ffc107', '#dc3545']
    
    fig = go.Figure(data=[go.Pie(
        labels=[s.title() for s in sentiments],
        values=sentiment_percentages,
        hole=0.4,  # Creates donut effect
        marker_colors=colors,
        textinfo='label+percent',
        textposition='auto'
    )])
    
    fig.update_layout(
        title=f"Sentiment Distribution Across {total_texts} Texts",
        height=400,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.1,
            xanchor="center",
            x=0.5
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

#  ====================== Main Application ======================

# Page configuration
st.set_page_config(
    page_title="Text Classification System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .model-card {
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
        background-color: #f8f9fa;
    }
    
    .prediction-result {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    .topic-positive { 
        background-color: #d4edda; 
        border-left: 4px solid #28a745; 
        color: #155724 !important;
    }
    .topic-neutral { 
        background-color: #fff3cd; 
        border-left: 4px solid #ffc107; 
        color: #856404 !important;
    }
    .topic-negative { 
        background-color: #f8d7da; 
        border-left: 4px solid #dc3545; 
        color: #721c24 !important;
    }
    .metrics-container {
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
    }
    
    .metric-box {
        text-align: center;
        padding: 1rem;
        border-radius: 8px;
        background-color: #ffffff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        min-width: 120px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 >🔍 Ecommerce Product Review Analysis - Indonesian Language </h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar for configuration
with st.sidebar:
    
    st.header("📊 Model Information")
    
    # Topic Model Info
    with st.expander("🏷️ Topic Classification Model", expanded=True):
        st.markdown("""
        **Model Type:** Multi-label Classification  
        **Categories:**
        - 📦 Product
        - 🎧 Customer Service  
        - 🚚 Shipping
        
        **Note:** Text can belong to multiple categories
        """)
    
    # Sentiment Model Info
    with st.expander("😊 Sentiment Analysis Model", expanded=True):
        st.markdown("""
        **Model Type:** Single-label Classification  
        **Categories:**
        - 😊 Positive
        - 😐 Neutral
        - 😞 Negative
        """)
    
    # Statistics (if available)
    if 'classification_history' in st.session_state:
        st.header("📈 Session Statistics")
        history = st.session_state.classification_history
        
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.metric("Texts Classified", len(history))
        with col_stat2:
            avg_confidence = np.mean([h['confidence'] for h in history])
            st.metric("Avg Confidence", f"{avg_confidence:.2f}")

# Import pipeline module with error handling
try:
    import pipeline
    # st.success("✅ Models loaded successfully!")
except ImportError as e:
    st.error(f"❌ Error importing pipeline module: {str(e)}")
    st.info("Please make sure your pipeline.py file is in the same directory and contains the required models.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading models: {str(e)}")
    st.stop()

# Main content
st.header("📝 Text Input")

# Input methods
input_method = st.radio("Choose input method:", ["Single Text", "Batch Upload", "Example Texts"])

if input_method == "Single Text":
    user_text = st.text_area(
        "Enter text to classify:",
        placeholder="Type or paste your text here...",
        height=150
    )
    
    if st.button("🚀 Classify Text", type="primary"):
        if user_text.strip():
            try:
                # Call your actual model prediction functions
                topic_predictions = simulate_topic_prediction(user_text)
                sentiment_prediction = simulate_sentiment_prediction(user_text)
                
                display_predictions(user_text, topic_predictions, sentiment_prediction)
            except Exception as e:
                st.error(f"Error during classification: {str(e)}")
        else:
            st.warning("Please enter some text to classify!")

elif input_method == "Batch Upload":
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        # Delimiter options
        col_delim, col_encoding = st.columns(2)
        with col_delim:
            delimiter = st.selectbox(
                "Select delimiter:",
                options=[",", ";", "\t", "|", " "],
                format_func=lambda x: {"," : "Comma (,)", ";" : "Semicolon (;)", "\t" : "Tab", "|" : "Pipe (|)", " " : "Space"}[x],
                index=0
            )
        
        with col_encoding:
            encoding = st.selectbox(
                "Select encoding:",
                options=["utf-8", "latin-1", "cp1252", "ascii"],
                index=0
            )
        
        try:
            df = pd.read_csv(uploaded_file, delimiter=delimiter, encoding=encoding)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())
            
            text_column = st.selectbox("Select text column:", df.columns)
            # maximum number of rows to process
            max_rows = st.slider(
                "Maximum rows to process:",
                min_value=1,
                max_value=len(df),
                value=min(100, len(df)),
                step=1
            )
            if st.button("🔄 Process Batch", type="primary"):
                process_batch_classification(df, text_column, max_results=max_rows)
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")
            st.info("Try different delimiter or encoding options if the file doesn't load correctly.")

else:  # Example Texts
    st.subheader("Try these example texts:")
    
    # Example type selection
    example_type = st.radio(
        "Choose example type:",
        ["Single Examples", "CSV Examples"],
        horizontal=True
    )
    
    if example_type == "Single Examples":
        examples = [
            "Pengiriman terlambat 3 hari dan paketnya rusak.",
            "Pelayanan pelanggan sangat baik! Tim support sangat membantu dan responsif.",
            "Kualitas produknya sangat bagus, sesuai dengan yang saya harapkan.",
            "Saya kesulitan dengan proses pengembalian barang, sangat membingungkan.",
            "Pengiriman cepat dan barang sampai dalam kondisi sempurna!"
        ]
        
        # Initialize session state for tracking which example to show results for
        if 'selected_example' not in st.session_state:
            st.session_state.selected_example = None
            st.session_state.example_results = None
        
        for i, example in enumerate(examples):
            col_ex1, col_ex2 = st.columns([4, 1])
            with col_ex1:
                st.text(f"{i+1}. {example}")
            with col_ex2:
                if st.button(f"Classify", key=f"example_{i}"):
                    try:
                        topic_predictions = simulate_topic_prediction(example)
                        sentiment_prediction = simulate_sentiment_prediction(example)
                        
                        # Store results in session state
                        st.session_state.selected_example = i
                        st.session_state.example_results = {
                            'text': example,
                            'topic_predictions': topic_predictions,
                            'sentiment_prediction': sentiment_prediction
                        }
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error during classification: {str(e)}")
        
        # Display results below all examples if any example was classified
        if st.session_state.selected_example is not None and st.session_state.example_results:
            results = st.session_state.example_results
            display_predictions(
                results['text'], 
                results['topic_predictions'], 
                results['sentiment_prediction']
            )
    
    
    else:  # CSV Examples
        st.markdown("**Pre-prepared CSV datasets for testing:**")
        
        # Predefined CSV options
        csv_options = {
            "Sample E-commerce Reviews": {
                "data": {
                    "review_text": [
                        "Produk bagus tapi pengiriman lama",
                        "Customer service tidak responsif",
                        "Barang sesuai deskripsi, packing aman",
                        "Pengiriman cepat tapi produk cacat",
                        "Pelayanan memuaskan, akan order lagi",
                        "Kualitas produk mengecewakan",
                        "Pengiriman sangat cepat dan aman",
                        "Tim support sangat membantu menyelesaikan masalah",
                        "Produk original dan sesuai gambar",
                        "Proses refund sangat lambat dan rumit"
                    ],
                    "rating": [4, 2, 5, 3, 5, 1, 5, 5, 4, 2],
                    "category": ["Electronics", "Fashion", "Books", "Electronics", "Fashion", "Electronics", "Books", "Fashion", "Electronics", "Fashion"]
                },
                "description": "Indonesian e-commerce reviews with mixed sentiments and topics"
            },
            "Product Reviews Dataset": {
                "data": {
                    "review_text": [
                        "Laptop ini performanya sangat bagus untuk gaming",
                        "Baju ini bahannya halus dan nyaman dipakai",
                        "Buku ini sangat informatif dan mudah dipahami",
                        "Handphone rusak setelah 2 minggu pemakaian",
                        "Sepatu ini sangat nyaman untuk jogging",
                        "Kamera foto hasil jelek, tidak sesuai harga",
                        "Pelayanan toko online ini sangat memuaskan",
                        "Pengiriman terlambat tapi barang aman",
                        "Produk tidak sesuai dengan deskripsi",
                        "Kualitas packaging sangat baik dan rapi"
                    ],
                    "product_type": ["Laptop", "Clothing", "Book", "Phone", "Shoes", "Camera", "Service", "Shipping", "General", "Packaging"],
                    "sentiment_label": ["positive", "positive", "positive", "negative", "positive", "negative", "positive", "neutral", "negative", "positive"]
                },
                "description": "Product-focused reviews with pre-labeled sentiments"
            },
            "Customer Service Reviews": {
                "data": {
                    "review_text": [
                        "CS sangat ramah dan membantu menyelesaikan komplain",
                        "Susah menghubungi customer service via telepon",
                        "Live chat responsive tapi solusi kurang tepat",
                        "Tim support email sangat profesional",
                        "Customer service tidak memberikan solusi yang jelas",
                        "Pelayanan 24/7 sangat membantu customer",
                        "CS galak dan tidak sabar melayani customer",
                        "Support ticket dijawab dengan cepat dan tepat"
                    ],
                    "channel": ["Phone", "Phone", "Chat", "Email", "Phone", "24/7", "Phone", "Ticket"],
                    "resolution": ["Resolved", "Unresolved", "Partial", "Resolved", "Unresolved", "Resolved", "Unresolved", "Resolved"]
                },
                "description": "Customer service specific reviews and interactions"
            }
        }
        
        # CSV selection
        selected_csv = st.selectbox(
            "Choose a pre-prepared dataset:",
            options=list(csv_options.keys()),
            help="Select from curated datasets for testing different scenarios"
        )
        
        if selected_csv:
            csv_info = csv_options[selected_csv]
            sample_df = pd.DataFrame(csv_info["data"])
            
            # Display info and preview
            st.info(f"📋 **{selected_csv}**: {csv_info['description']}")
            
            col_preview, col_actions = st.columns([3, 1])
            
            with col_preview:
                st.dataframe(sample_df, use_container_width=True)
            
            with col_actions:
                # Download button
                csv_data = sample_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name=f"{selected_csv.lower().replace(' ', '_')}.csv",
                    mime="text/csv",
                    help="Download this dataset to test batch processing"
                )
                
                # Quick test button
                if st.button("🚀 Quick Test", help="Automatically process this dataset"):
                    st.session_state['quick_test_df'] = sample_df
                    st.session_state['quick_test_column'] = 'review_text'
                    st.rerun()
        
        # Handle quick test
        if 'quick_test_df' in st.session_state:
            st.markdown("---")
            st.subheader("🔄 Quick Test Results")
            process_batch_classification(
                st.session_state['quick_test_df'], 
                st.session_state['quick_test_column'], 
                len(st.session_state['quick_test_df'])
            )
            # Clear session state
            del st.session_state['quick_test_df']
            del st.session_state['quick_test_column']
        
        st.info("💡 **Tip:** Download any dataset above and upload it in the 'Batch Upload' section, or use 'Quick Test' for immediate processing!")

# Footer
st.markdown("---")