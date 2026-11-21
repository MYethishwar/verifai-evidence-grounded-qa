import re
import string
from datetime import datetime

def preprocess_text(text):
    """Clean and normalize biomedical text preserving readable capitalization."""
    if not text or not isinstance(text, str):
        return ''
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove special chars but keep medical punctuation
    text = re.sub(r'[^\w\s\-\.\,\;\:\(\)\/]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def segment_passages(text, max_words=150):
    """Divide text into passages of max_words."""
    if not text:
        return []
    sentences = re.split(r'(?<=[.!?])\s+', text)
    passages = []
    current = []
    word_count = 0
    for sent in sentences:
        words = sent.split()
        if word_count + len(words) > max_words and current:
            passages.append(' '.join(current))
            current = [sent]
            word_count = len(words)
        else:
            current.append(sent)
            word_count += len(words)
    if current:
        passages.append(' '.join(current))
    return [p for p in passages if len(p.split()) >= 5]

def extract_keywords(text, top_n=10):
    """Extract top keywords from text using frequency."""
    if not text:
        return []
    stop_words = set(['the','a','an','and','or','but','in','on','at','to','for',
                      'of','with','by','from','is','are','was','were','be','been',
                      'have','has','had','do','does','did','will','would','could',
                      'should','may','might','this','that','these','those','it',
                      'its','we','they','their','our','you','your','not','no'])
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    freq = {}
    for w in words:
        if w not in stop_words:
            freq[w] = freq.get(w, 0) + 1
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in sorted_words[:top_n]]

def preprocess_dataframe(df):
    """Preprocess a pandas DataFrame of biomedical data."""
    import pandas as pd
    # Drop duplicates
    df = df.drop_duplicates()
    # Fill NaN
    df = df.fillna('')
    # Detect columns
    text_columns = []
    for col in df.columns:
        col_lower = col.lower()
        if any(k in col_lower for k in ['abstract', 'text', 'content', 'body', 'description', 'title', 'answer']):
            text_columns.append(col)
    for col in text_columns:
        df[col] = df[col].apply(lambda x: preprocess_text(str(x)))
    return df, text_columns

def get_preprocessing_stats(df):
    """Return stats about the dataframe."""
    return {
        'total_records': len(df),
        'columns': list(df.columns),
        'missing_values': int(df.isnull().sum().sum()),
        'duplicate_rows': 0
    }
