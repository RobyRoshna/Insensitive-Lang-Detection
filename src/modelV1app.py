import streamlit as st
from transformers import BertForSequenceClassification, BertTokenizer
import torch

# Load model and tokenizer from Hugging Face
@st.cache_resource()
def load_model():
    model = BertForSequenceClassification.from_pretrained("rrroby/insensitive-language-bert")
    tokenizer = BertTokenizer.from_pretrained("rrroby/insensitive-language-bert")
    return model, tokenizer

model, tokenizer = load_model()

# App title and description
st.title("Disability Insensitive Language Detection Version 1")
st.write(
    """
    Paste your abstract or academic text below. 
    It will be analyzed and flagged if any disability-insensitive language is detected.
    NOTE: Accuracy is limited as the current model was trained on very little data
    """
)

# User input box
text = st.text_area("Enter text here:", height=300)

# Analyze button
if st.button("Analyze"):
    if text.strip() == "":
        st.warning("Some text required for analysis")
    else:
        # Tokenize input
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred_class = torch.argmax(probs, dim=-1).item()

        # Show results
        if pred_class == 1:
            st.error("Insensitive language detected!")
        else:
            st.success("No insensitive language detected.")

        st.write(f"**Model confidence (probabilities):** {probs.tolist()[0]}")
