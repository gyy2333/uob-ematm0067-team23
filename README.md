# uob-ematm0067-team23

AI Text Analytics Project: Preprocessing, Topic Modeling, and Trend Analysis

## Overview
This project analyses a large corpus of scientific abstracts to understand how research focus, framing, and communication have evolved over time in the fields of Artificial Intelligence (AI), Machine Learning (ML), and Natural Language Processing (NLP).
The analysis applies multiple text analytics techniques including TF-IDF, n-grams, embeddings, classification, Keyword analysis, and PCA to extract insights and compare different analytical approaches.
### Objective
•	Analyse how research topics and terminology have changed over time
•	Compare AI, ML, and NLP subfields
•	Evaluate how different text analysis methods affect insights
•	Identify temporal drift in research using dimensionality reduction
### Preprocessing steps:
•	Extracted year from id
•	Created time periods:
o	1991–1998
o	1999–2006
o	2007–2014
o	2015–2021
•	Filtered dataset to focus on: ['cs.AI', 'cs.LG', 'cs.CL']
### Methodology
 1. Data Cleaning
•	Removed unnecessary columns
•	Handled missing values
•	Standardised category format
2. Category Selection
•	Focused on AI, ML, and NLP-related subcategories
•	Reduced noise from irrelevant fields
3. Data Imbalance Handling
•	Identified strong imbalance across time periods
•	Used normalisation during analysis to ensure fair comparison
 4. Text Preprocessing
•	Lowercasing
•	Removing special characters
•	Tokenization
•	Stopword removal
•	Lemmatization
5. Text Representation Methods
TF-IDF
•	Captures importance of words
•	Highly interpretable
N-grams
•	Captures phrases (used ngrams-(2,3))
•	Improves contextual understanding
Embeddings
•	Captures semantic meaning
•	Enables similarity-based analysis
6. Analysis Methods
Keyword Analysis
•	Extracted top words globally, by time, and by category
•	Used to identify trends and research focus
Classification
•	Predicted time period from text
•	Measures how much language changes over time
PCA (Principal Component Analysis)
•	Reduced high-dimensional TF-IDF vectors to 2D
•	Visualised temporal drift and research evolution

## Run
```{bash}
cd uob-ematm0067-team23
bash src/run.sh
```