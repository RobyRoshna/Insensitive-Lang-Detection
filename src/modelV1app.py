import streamlit as st
from transformers import BertForSequenceClassification, BertTokenizer
import torch
import nltk

# Download sentence tokenizer data
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# Load model and tokenizer
@st.cache_resource()
def load_model():
    model = BertForSequenceClassification.from_pretrained("rrroby/insensitive-language-bert")
    tokenizer = BertTokenizer.from_pretrained("rrroby/insensitive-language-bert")
    return model, tokenizer

model, tokenizer = load_model()

# Page title and instructions
st.title("Disability Insensitive Language Detection V1.2")
st.write(
    """
    Paste your abstract or academic text below.
    Each sentence will be analyzed and flagged if any disability-insensitive language is detected.\n
    NOTE: The current model was trained on very little data and is still in the early stages, therefore, it is prone to inaccuracies.
    """
)

text = st.text_area("Enter text here:", height=250)

if st.button("Analyze"):
    if text.strip() == "":
        st.warning("Some text required for analysis")
    else:
        sentences = sent_tokenize(text)

        with st.spinner("Analyzing..."):
            inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                pred_classes = torch.argmax(probs, dim=-1)

        for idx, sentence in enumerate(sentences):
            prob_not_insensitive = probs[idx][0].item() * 100
            prob_insensitive = probs[idx][1].item() * 100

            if pred_classes[idx] == 1:
                st.error(f"**Insensitive:** {sentence}")
            else:
                st.success(f"**Not insensitive:** {sentence}")

            st.caption(f"Model's Confidence — Not insensitive: {prob_not_insensitive:.2f}%, Insensitive: {prob_insensitive:.2f}%")
