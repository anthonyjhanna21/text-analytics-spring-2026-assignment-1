Task: Text Cleaning Functions,   Date: 2026-02-5,   Time spent: 15 minutes
---

What I was trying to do:
Build reusable Python text-cleaning functions for my sentiment analysis project, including removing HTML tags, URLs, rating patterns, special characters, and extra whitespace.

AI Prompt:
how do I remove urls from text in python with regex quick

AI Response:
import re
clean = re.sub(r'http\S+|www.\S+
', '', text)

What I Modified:
Adjusted the URL pattern slightly in my function to replace matches with a space instead of empty string so words would not get stuck together. Put the regex inside a reusable remove_urls() function.

Why I Modified:
Direct removal sometimes merged neighboring words. Replacing with a space kept token separation cleaner for later NLP steps.

What I Learned:
Regex patterns like http\S+ and www.\S+
 capture most links. Wrapping cleaning steps into functions makes preprocessing consistent across the dataset.

AI Errors Found:
Initial example did not mention spacing side effects after removal, which I corrected by inserting spaces.

AI Prompt:
fast way to remove html tags like br from text python regex?

AI Response:
clean = re.sub(r'<[^>]+>', '', text)

What I Modified:
Used a slightly different regex pattern r"<.*?>" in my remove_html() function and replaced tags with a space instead of empty string.

Why I Modified:
Using a space kept sentence structure readable and prevented word collisions after tag removal.

What I Learned:
Angle-bracket regex removes most simple HTML tags. Regex is okay for quick cleaning but not perfect for complex HTML.

AI Errors Found:
AI suggested BeautifulSoup as an alternative but I chose regex for speed and simplicity since my dataset text is short reviews.

AI Prompt:
regex to remove stuff like 8/10 or 10/10 from text?

AI Response:
clean = re.sub(r'\b\d+/\d+\b', '', text)

What I Modified:
I handled numeric and symbol cleanup more generally using a special-character removal regex that strips non letter characters.

Why I Modified:
My broader special-character filter already removed slashes and numbers

What I Learned:
Multiple regex steps can overlap

AI Errors Found:
No direct errors, but the suggested step was redundant with my later special-character cleaning function.

AI Prompt:
how to strip special characters but keep letters and spaces python

AI Response:
re.sub(r'[^a-zA-Z\s]', ' ', text)

What I Modified:
Placed this into a remove_special_chars() function

Why I Modified:
I needed a reusable function

What I Learned:
Cleaning steps should be ordered to avoid repeated noise.

AI Errors Found:
AI example did not include whitespace normalization, which I added separately.




Task: Build clean_text pipeline and edge case checks
Date: 2026-02-5
Time spent: 20 minutes
---
What I was trying to do:
Create one cleaning function to apply to my dataframe

AI Prompt:
help me get started making a clean_text() function that runs my cleaning steps in order and then apply it to a pandas column

AI Response:
Guidance: Put all steps in one clean_text(text) function, apply it with df["review"].apply(clean_text), then validate with quick counts (empty strings, short word counts, non-ascii flag, and raw HTML/URL contains checks).

What I Modified:
I added a rating regex step and summary counts for empty and very short cleaned reviews

Why I Modified:
I needed consistent preprocessing before sentiment scoring

What I Learned:
Simple count checks make it easy to catch broken preprocessing early.

AI Errors Found:
None








Task: Before/after cleaning spot-check helper
Date: 2026-02-5
Time spent: 10 minutes
---
What I was trying to do:
Create a quick way to visually compare raw vs cleaned reviews

AI Prompt:
start writing a debug helper that prints before and after text from a dataframe for a few sampled rows that match patterns like html, urls, or weird characters

AI Response:
You can make a helper that loops through a list of indices and prints the original and cleaned columns for each one. Use df.loc with the index and column names so you always get the correct row.

Example structure close to what you want:

def show_before_after(idxs):
for idx in idxs:
before_text = df.loc[idx, "review"]
after_text = df.loc[idx, "clean_review"]
print("Row:", idx)
print("Before:", before_text)
print("After:", after_text)
print("-"*80)

To get example indices, filter first, then sample:

html_idx = df[df["review"].astype(str).str.contains(r"<.*?>", regex=True)]
.sample(1).index.tolist()

url_idx = df[df["review"].astype(str).str.contains(r"http\S+|www\S+", regex=True)]
.sample(1).index.tolist()

For non-ASCII characters, use a lambda check and sample from that filtered set. Then combine the index lists and pass them into your function.

What I Modified:
Printed formatting, added fixed random_state sampling, and built three separate filtered samples before combining.

Why I Modified:
I wanted reproducible samples and clearer labeled output.

What I Learned:
Filtering, sampling, and a small print helper is a good method to validate preprocessing results.







Task: VADER and TextBlob sentiment pipelines
Date: 2026-02-10
Time spent: 30 minutes
---
What I was trying to do:
Set up VADER and TextBlob sentiment scoring. Create small pipeline functions so I could test sentiment outputs on a few sample reviews.


AI Prompt:
help me get started setting up vader and textblob sentiment in python and making small pipeline functions for each so i can test a few sample rows

AI Response:
To set up VADER, download the lexicon once, then create a SentimentIntensityAnalyzer instance and wrap it in a small pipeline function. It helps to preprocess the text first (remove HTML, URLs, extra spaces) before scoring.

Example structure:

import nltk
nltk.download("vader_lexicon")

from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

vader = SentimentIntensityAnalyzer()

def vader_clean(t):
t = str(t)
t = re.sub(r"<.*?>", " ", t)
t = re.sub(r"http\S+|www\S+", " ", t)
return re.sub(r"\s+", " ", t).strip()

def vader_score(t):
t = vader_clean(t)
return vader.polarity_scores(t)

For TextBlob, you can build a small wrapper that returns polarity and subjectivity:

def blob_score(t):
t = str(t)
b = TextBlob(t)
return {"pol": b.sentiment.polarity,
"subj": b.sentiment.subjectivity}

To sanity check, sample a few dataframe rows and print raw text plus both model outputs using df.sample() and a short loop.

What I Modified:
Renamed functions to match my pipeline, separated vader_preprocess and vader_pipeline, and aligned outputs to dictionary format.

Why I Modified:
I wanted consistent preprocessing and clean reusable scoring functions for each model.

What I Learned:
Lexicon models still need light preprocessing and are easier to test with small dataframe samples first.



Task: DistilBERT emotion model pipeline
Date: 2026-02-10
Time spent: 30 minutes
---
What I was trying to do:
Load a pretrained DistilBERT emotion model and build a function that converts text into emotion probability scores for sampled reviews.

AI Prompt:
help me get started running a pretrained distilbert emotion model in python and wrapping it in a function that takes text and returns class probabilities

AI Response:
%pip install transformers torch

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "bhadresh-savani/distilbert-base-uncased-emotion"
tok = AutoTokenizer.from_pretrained(model_name)
clf = AutoModelForSequenceClassification.from_pretrained(model_name)

label_names = list(clf.config.id2label.values())

def emotion_scores(t, max_len=256):
t = str(t).strip()
if not t:
return {lab: 0.0 for lab in label_names}

enc = tok(
    t,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=max_len
)

with torch.no_grad():
    out = clf(**enc)
    prob_vec = torch.softmax(out.logits, dim=1)[0]

return dict(zip(label_names, prob_vec.tolist()))

What I Modified:
I matched variable names to my notebook, added empty-text handling, and converted outputs to float values.

Why I Modified:
I wanted safe handling for blank reviews. I also wanted consistent dictionary outputs for dataframe expansion.

What I Learned:
Transformer models require tokenization and softmax on logits. Wrapping inference in a function makes dataframe apply testing easier.

AI Errors Found:
Example used different variable names; I aligned them with my existing dataframe and pipeline.







Task: Edge-case sentiment comparison table
Date: 2026-02-15
Time spent: 15 minutes
---
What I was trying to do:
Test my VADER, TextBlob, and DistilBERT pipelines on a few manual edge-case strings and combine all outputs into one dataframe.

AI Prompt:
help me get started testing my sentiment pipelines on a few manual edge case strings and putting all model outputs into one dataframe row per case

AI Response:
Example structure close to your goal:

edge_cases = {
"empty": "",
"non_english": "Texto de prueba aquí",
"emoji_text": "Amazing movie!!! 😄🔥 http://test.com
"
}

rows = []

for label, text in edge_cases.items():
v = vader_pipeline(text)
b = textblob_pipeline(text)
d = distilbert_emotion_scores(text)

row = {
    "case": label,
    "text": text,
    "vader_compound": v["compound"],
    "blob_polarity": b["polarity"],
    "blob_subjectivity": b["subjectivity"]
}

row.update(d)
rows.append(row)


edge_df = pd.DataFrame(rows)

This lets you compare how each model behaves on tricky inputs and keeps one row per edge case.

What I Modified:
Used my existing scoring function names, added specific edge-case examples, and expanded each model’s scores into the result rows.

Why I Modified:
I wanted a compact comparison table showing how each model reacts to unusual inputs.

What I Learned:
Looping through a labeled edge-case dictionary is a good pattern for multi-model testing.









Task: Convert model scores to sentiment labels
Date: 2026-02-15
Time spent: 25 minutes
---
What I was trying to do:
Turn VADER, TextBlob, and DistilBERT score outputs into positive/negative/neutral labels and apply them to a sampled dataframe.

AI Prompt:
how do I convert sentiment model scores into positive negative neutral labels and applying them to a pandas sample dataframe

AI Response:
A common pattern is to write a small labeling function per model that converts its numeric score into a category using thresholds, then apply that function to a dataframe column.

For VADER, use the compound score and standard cutoffs around ±0.05:

def vader_label(t):
score = vader_pipeline(t)["compound"]
if score >= 0.05:
return "positive"
elif score <= -0.05:
return "negative"
return "neutral"

For TextBlob, use polarity with slightly wider thresholds like ±0.1 to avoid weak signals:

def blob_label(t):
pol = textblob_pipeline(t)["polarity"]
if pol > 0.1:
return "positive"
elif pol < -0.1:
return "negative"
return "neutral"

For an emotion model, take the top-scoring class and map groups of emotions into positive/negative buckets:

pos_set = {"joy","love"}
neg_set = {"anger","sadness","fear"}

def bert_sentiment(t):
sc = distilbert_emotion_scores(t)
top_label = max(sc, key=sc.get)
if top_label in pos_set:
return "positive"
elif top_label in neg_set:
return "negative"
return "neutral"

Then create a sampled dataframe and add new label columns with .apply() for each function.

What I Modified:
Used my own function names, thresholds, and emotion group sets, and applied to specific columns (raw vs cleaned text).

Why I Modified:
I wanted consistent three-class labels across all models for comparison.

What I Learned:
Simple wrapper label functions plus apply() make multi-model comparison easy.
