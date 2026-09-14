import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))


def preprocess_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    tokens = word_tokenize(text)

    cleaned_tokens = [
        stemmer.stem(token)
        for token in tokens
        if token.isalpha() and token not in stop_words
    ]

    return " ".join(cleaned_tokens)


def main():
    df = pd.read_csv("data/clean_issues.csv")
    print("Preprocessing text with NLTK...")
    df["clean_text"] = df["text"].apply(preprocess_text)

    # Filter out any issues that might have become empty after cleaning
    df = df[df["clean_text"].str.strip() != ""]

    output_path = "data/preprocessed_issues.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved preprocessed data to {output_path} ({len(df)} rows)")

    print("\nSample before and after:")
    sample = df.iloc[0]
    print("BEFORE:\n", sample["text"][:150], "...")
    print("AFTER:\n", sample["clean_text"][:150], "...")


if __name__ == "__main__":
    main()

