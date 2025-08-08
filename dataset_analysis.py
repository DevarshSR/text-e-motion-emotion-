import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources (run only once)
nltk.download('punkt')
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("data/train.txt", sep=";", header=None, names=["text", "emotion"])

# Lowercase text
df["text_clean"] = df["text"].str.lower()

# Remove punctuation
df["text_clean"] = df["text_clean"].apply(lambda x: x.translate(str.maketrans("", "", string.punctuation)))

# Tokenize text
df["tokens"] = df["text_clean"].apply(word_tokenize)

# Remove stopwords
stop_words = set(stopwords.words('english'))
df["tokens"] = df["tokens"].apply(lambda x: [word for word in x if word not in stop_words])

# Show the cleaned result
print("\n--- Cleaned Tokens Example ---")
print(df[["text", "tokens"]].head(10))
