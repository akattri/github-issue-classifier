import csv
import os
import time
import requests
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import pandas as pd

SEARCH_URL = "https://api.github.com/search/issues"
TOKEN = os.getenv("GITHUB_TOKEN")

HEADERS = {"Accept": "application/vnd.github+json"}
if TOKEN:
    HEADERS["Authorization"] = f"Bearer {TOKEN}"

QUERIES = [
    ("is:issue label:security", "SECURITY"),
    ("is:issue label:vulnerability", "SECURITY"),
    ("is:issue label:documentation", "NON_SECURITY"),
    ("is:issue label:enhancement", "NON_SECURITY"),
]


def fetch_issues_for_query(query, target_label, count=50):
    params = {
        "q": query,
        "per_page": count,
    }
    print(f"Searching: '{query}' -> {target_label}...")
    response = requests.get(SEARCH_URL, headers=HEADERS, params=params)

    if response.status_code == 403:
        print("Rate limit reached. Set GITHUB_TOKEN if needed.")
        return []
    elif response.status_code != 200:
        print(f"Search failed: {response.status_code} {response.text}")
        return []

    items = response.json().get("items", [])
    collected = []
    for item in items:
        title = item.get("title", "") or ""
        body = item.get("body", "") or ""

        if not title.strip() and not body.strip():
            continue

        collected.append({
            "repo": item.get("repository_url", "").replace("https://api.github.com/repos/", ""),
            "number": item.get("number"),
            "title": title.replace("\n", " ").strip(),
            "body": body.replace("\r\n", "\n").strip(),
            "label": target_label,
        })

    time.sleep(1.0)
    return collected

def clean_data():
    df=pd.read_csv("data/raw_issues.csv")
    df["body"] = df["body"].fillna("")
    df["text"] = df["title"] + " " + df["body"]
    df["text"] = df["text"].str.strip()
    df = df[df["text"] != ""]
    df = df.drop_duplicates(subset=["text"])
    clean_df = df[["text", "label"]]
    clean_df.to_csv("data/clean_issues.csv", index=False)
    print(f"Cleaned dataset saved: {len(clean_df)} issues.")
    print(clean_df["label"].value_counts())



def main():
    all_issues = []
    for query, label in QUERIES:
        results = fetch_issues_for_query(query, label, count=50)
        print(f"Collected {len(results)} issues for {label}")
        all_issues.extend(results)

    output_path = os.path.join("data", "raw_issues.csv")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["repo", "number", "title", "body", "label"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_issues)

    print(f"\nDone! Saved {len(all_issues)} issues to {output_path}")
    clean_data()


if __name__ == "__main__":
    main()

