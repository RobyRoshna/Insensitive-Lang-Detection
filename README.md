
# Identifying Disability Insensitive Language in Scholarly Works using Machine Learning

##  Live Demo
[Click here to try the model on Hugging Face Spaces as a Streamlit App](https://huggingface.co/spaces/rrroby/Insensitive_Lang_DetectionV1)

---

## Overview
This repository contains the research and code for a project aimed at detecting and analyzing insensitive language concerning disability. Using machine learning techniques, this project identifies phrases that deviate from inclusive language guidelines. The ultimate goal is to provide insights and tools that promote more respectful communication informed by the social models of disability.

---


## Objectives
- Compile a list of guidelines and lexicons around insensitivity in disability-related language, specifically within written communication.
- Extract sentences that use terms or phrases referred to by the guidelines within the ASSETS conference papers over the years from the very first conference in 1994 to 2024, the latest one as per the date of this study.
- Study the potential of using OpenAI’s GPT-4o supported data augmentation where the corpora lacks sufficient examples.
- Predict insensitivity in disability language used within academia using a fine-tuned BERT model.

---


## Data Sources
- **Organizations**: ADA National Network, UN guidelines, and ASSETS

---

## Methodology
1. **Data Collection**: Data extraction from scholarly and organizational sources.
2. **Annotation**: Manual annotation for reliability.
3. **Keyword Extraction**: Identify terms and phrases relevant to disability.
4. **Model Training**:
   - Use natural language processing (NLP) techniques (BERT) for semantic analysis.
   - Compare with traditional logistic regression
5. **Evaluation**: Validate model accuracy using annotated datasets and comparison against inclusive guidelines.

---
## Contact
1. **Author**: Roshna Rebaca Roby (rrroby@upei.ca, roshnarebaca2002@gmail.com)
2. **Supervisors**: Christopher Power, Paul Sheridan
3. **University**: University of Prince Edward Island
---
## Important Note
If you are using Google Colab as in this project, ensure to modify the file paths to match your Google Drive paths. For example:
```python
from google.colab import drive
drive.mount('/content/drive') # Replace paths in scripts to use '/content/drive/My Drive/...' structure.
```
