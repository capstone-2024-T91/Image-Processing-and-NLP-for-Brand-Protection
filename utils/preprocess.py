import re

def preprocess_email(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove special characters
    text = re.sub(r'[^A-Za-z0-9\s]+', '', text)
    # Convert to lowercase
    text = text.lower()
    return text
