# Identifying Insensitive Language about Disabled People Using Semantic Analysis and Machine Learning

## Overview
This repository contains the research and code for a project aimed at detecting and analyzing insensitive language concerning disability. Using semantic analysis and machine learning techniques, this project identifies phrases that deviate from inclusive language guidelines. The ultimate goal is to provide insights and tools that promote more respectful communication informed by the medical and social models of disability.

---

## Objectives
- **Analyze Disability-Related Terminology**: Study the terms used to refer to disability across different document types (e.g., academic papers, charitable organizations).
- **Compare Language Styles**: Compare these terms with inclusive language standards as outlined by organizations like the ADA National Network and the UN Guidelines.
- **Detect Insensitive Language**: Develop a machine learning-based semantic analysis model to detect potentially insensitive language and provide recommendations for inclusive alternatives.
- **Raise Awareness**: Highlight where inclusivity may be lacking and inform tools for inclusive writing and communication, especially within the academic context.

---

## Features
- **Language Analysis**: Detect frequency and sentiment of disability-related terms in text.
- **Machine Learning Models**: Train a BERT model using custom annotated datasets for semantic analysis.
- **Guidelines Comparison**: Compare terms with inclusive language recommendations from resources like UN Disability-Inclusive Language Guidelines.

---

## Data Sources
- **Charitable Organizations**: Websites like ADA National Network.
- **Research Conferences**: Papers from ACM CHI and ASSETS
- **Public Text Data**: Open datasets related to disability and hate speech (e.g., Twitter datasets) for reference.

---

## Methodology
1. **Data Collection**: Web scraping and data extraction from scholarly and organizational sources.
2. **Annotation**: Combine manual and automated approaches to label texts for inclusivity.
3. **Keyword Extraction**: Identify terms and phrases relevant to disability.
4. **Model Training**:
   - Use natural language processing (NLP) techniques (BERT, SVM) for semantic analysis.
5. **Evaluation**: Validate model accuracy using annotated datasets and comparison against inclusive guidelines.

---
## Contact
Author: Roshna Rebaca Roby
Supervisors: Christopher Power, Paul Sheridan
University: University of Prince Edward Island
---
## Important Note
If you are using Google Colab as in this project, ensure to modify the file paths to match your Google Drive paths. For example:
```python
from google.colab import drive
drive.mount('/content/drive') # Replace paths in scripts to use '/content/drive/My Drive/...' structure.
```
## Repository Structure
```plaintext
├── data/
│   ├── raw/                   # Raw datasets
│   ├── processed/             # Processed datasets for modeling
├── models/
│   ├── bert/                  # Pretrained BERT models
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   ├── BERTtraining.ipynb
├── src/
│   ├── data_processing.py     # Data cleaning and preprocessing scripts
├── tests/
│   ├── test_data_processing.py
│   ├── test_model.py
├── docs/
│   ├── inclusive_language_guidelines.md
│   ├── research_summary.md
└── README.md                  # Project description
