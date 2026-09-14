import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix


def train_and_evaluate():
    df = pd.read_csv("data/preprocessed_issues.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    print(f"Training on {len(X_train)} issues, testing on {len(X_test)} issues.")

    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)

    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=["NON_SECURITY", "SECURITY"])

    print("\n=== CLASSIFICATION REPORT ===")
    print(report)
    print("=== CONFUSION MATRIX (labels: [NON_SECURITY, SECURITY]) ===")
    print(cm)

    # Show top learned words for each class
    feature_names = vectorizer.get_feature_names_out()
    for i, class_label in enumerate(model.classes_):
        top_indices = model.feature_log_prob_[i].argsort()[-10:][::-1]
        top_words = [feature_names[idx] for idx in top_indices]
        print(f"\nTop indicative words for {class_label}:")
        print(", ".join(top_words))

    os.makedirs("results", exist_ok=True)
    with open("data/model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("data/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    with open("results/metrics.txt", "w") as f:
        f.write("=== CLASSIFICATION REPORT ===\n")
        f.write(report)
        f.write("\n=== CONFUSION MATRIX ===\n")
        f.write(str(cm))

    print("\nModel saved to data/model.pkl")
    print("Metrics saved to results/metrics.txt")


if __name__ == "__main__":
    train_and_evaluate()

