import torch
import numpy as np
from transformers import BertModel, AutoTokenizer
from model_class import CustomClassifierAspect, CustomClassifierSentiment
import streamlit as st

ready_status = False
bert = None
tokenizer = None
aspect_model = None
sentiment_model = None


with st.status("Loading models...", expanded=True, state='running') as status:
    # Load the base model and tokenizer
    bertAspect = BertModel.from_pretrained("indobenchmark/indobert-base-p1",
                                                            num_labels=3,
                                                           problem_type="multi_label_classification")
    bertSentiment = BertModel.from_pretrained("indobenchmark/indobert-base-p1")
    
    tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")

    # Load custom models
    aspect_model = CustomClassifierAspect.from_pretrained("fahrendrakhoirul/indobert-finetuned-ecommerce-product-reviews-aspect-multilabel", bert=bertAspect)
    sentiment_model = CustomClassifierSentiment.from_pretrained("fahrendrakhoirul/indobert-finetuned-ecommerce-product-reviews-sentiment", bert=bertSentiment)
    st.write("Model loaded")


    # Update status to indicate models are ready
    if aspect_model and sentiment_model  != None:
        ready_status = True
    if ready_status:
        status.update(label="Models loaded successfully", expanded=False)
        status.success("Models loaded successfully", icon="✅")
    else:
        status.error("Failed to load models")