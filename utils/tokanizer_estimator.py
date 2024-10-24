import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.data_loader import load_pure_dataset
import tiktoken
from tqdm import tqdm
import matplotlib.pyplot as plt

tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
print(f"Tokenizer: {tokenizer}")
# Function to calculate the number of tokens in a text
def count_tokens(text):
  return len(tokenizer.encode(text))

# Load the dataset
texts, labels = load_pure_dataset(1)
print(f"Number of texts in the dataset: {len(texts)}")
# If texts is a list of strings, you cannot use ['body'], just use texts directly
texts_with_tokens = [{'text': text, 'num_tokens': count_tokens(text)} for text in tqdm(texts, desc="Counting tokens")]

# Convert to DataFrame to plot
import pandas as pd
texts_df = pd.DataFrame(texts_with_tokens)

print(f"Number of tokens in 5% of the dataset: {texts_df['num_tokens'].sum()}")

