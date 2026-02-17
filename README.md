# Sentiment Analysis Comparison Project

## Project Overview
This project performs sentiment analysis on a movie review text dataset using three different pretrained approaches: **VADER**, **TextBlob**, and a **Transformer model (DistilBERT emotion classifier)**.  
The goal is to compare how rule-based, lexicon-based, and transformer-based models differ in preprocessing needs, sentiment scoring, accuracy, and behavior on real reviews.

The project walks through full text analytics steps: data exploration, cleaning, feature preparation, model application, manual labeling, and model comparison.

---

## Dataset Description
- Dataset: Movie review text dataset (50,000 reviews)
- Main column used: `review`
- Type: Long-form English text reviews
- Size: ~50,000 rows
- Contains:
  - HTML tags
  - Some URLs
  - Special characters and punctuation
  - Mixed sentiment and long multi-sentence opinions

EDA steps included:
- Column and datatype inspection
- Missing value checks
- Text length statistics
- Text length histogram
- Sample review inspection

---

## Cleaning & Preprocessing
A custom cleaning pipeline was built that:
- Removes HTML tags
- Removes URLs
- Expands common contractions
- Removes special characters and numbers
- Normalizes whitespace
- Lowercases text (for TextBlob pipeline)

Important difference:
- **VADER used light preprocessing only** (kept punctuation and capitalization signals).
- **TextBlob used fully cleaned text.**
- **Transformer used near-raw text** with minimal cleaning.

---

## Models Used

### VADER
- Rule-based sentiment analyzer
- Outputs: negative, neutral, positive, compound score
- Strong with punctuation and emphasis

### TextBlob
- Lexicon-based sentiment analyzer
- Outputs: polarity and subjectivity
- Uses standard tokenization and averages word sentiment

### Transformer — DistilBERT Emotion Model
- Model: distilbert-base-uncased-emotion
- Outputs emotion probabilities:
  - joy, love, surprise, sadness, anger, fear
- Mapped to:
  - Positive → joy/love/surprise
  - Negative → sadness/anger/fear
  - Neutral → fallback

---

## Key Findings Summary

- Transformer model achieved the **best overall accuracy** on the manually labeled 100-review sample.
- VADER was **fastest** and handled punctuation and emphasis best.
- TextBlob was simplest but weakest on complex or mixed reviews.
- Transformer handled:
  - Negation
  - Context
  - Long reviews
  - Mixed sentiment
  better than the other models.

Example strengths:
- VADER → caps + !!! emphasis
- TextBlob → simple polarity detection
- Transformer → contextual meaning and sequence understanding

---

## Model Comparison (High Level)

| Criterion | Winner |
|------------|-----------|
Speed | VADER |
Accuracy | Transformer |
Handles emphasis | VADER |
Handles negation | Transformer |
Long reviews | Transformer |

---

## How To Run

1. Open the notebook:
Assignment_1.ipynb

2. Install required packages if needed:
```python
pip install vaderSentiment textblob transformers torch
```

3. Run cells in order:
Data loading
EDA
Cleaning pipeline
Model pipelines
Step 5 model application
Step 6 evaluation & comparison

4. For accuracy section:
Export 100-sample file
Manually label ground_truth column
Re-import labeled file
Run accuracy report cells

Results & Visualizations

The notebook includes:
Text length histogram
Descriptive statistics
Before/after cleaning examples
Model score outputs
Accuracy reports
Success and failure case reviews
Model comparison table

Files Included: 
Assignment_1.ipynb: main notebook
https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews: Dataset used for sentiment analysis (50,000 movie reviews)

Final Recommendation

For this dataset, the Transformer model is recommended for best accuracy and contextual understanding.
VADER is best when speed and simplicity are required.
TextBlob is useful as a lightweight baseline.
README.md — project documentation
