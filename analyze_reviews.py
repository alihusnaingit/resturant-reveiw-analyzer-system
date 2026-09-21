"""
Local review sentiment analyzer.

Loads your locally-downloaded fine-tuned model and classifies a batch of
customer reviews, then prints a positive/negative breakdown.

Usage:
    python analyze_reviews.py reviews.csv
    (CSV must have a column named "review")

    or, with no CSV, it runs on the SAMPLE_REVIEWS list below so you can
    test it immediately without preparing a file.
"""

import sys
from transformers import pipeline

MODEL_PATH = "."  # change to wherever you unzipped it

SAMPLE_REVIEWS = [
    "The food tastes amazing, the chicken tikka is awesome",
    "Service was slow and the pizza was cold",
    "Decent place, nothing special but not bad either",
    "Worst meal I've had in years, sent it back",
    "Best butter chicken in town, will come back every week",
    "Waited 40 minutes and the order was still wrong",
    "Loved the ambiance and the staff was super friendly",
    "Overpriced for the portion size, disappointed",
]


def load_reviews_from_csv(path: str):
    import pandas as pd
    df = pd.read_csv(path)
    if "review" not in df.columns:
        raise ValueError("CSV must have a column named 'review'")
    return df["review"].astype(str).tolist()


def main():
    if len(sys.argv) > 1:
        reviews = load_reviews_from_csv(sys.argv[1])
        print(f"Loaded {len(reviews)} reviews from {sys.argv[1]}\n")
    else:
        reviews = SAMPLE_REVIEWS
        print(f"No CSV given — running on {len(reviews)} sample reviews\n")

    classifier = pipeline("sentiment-analysis", model=MODEL_PATH, tokenizer=MODEL_PATH)

    results = classifier(reviews)

    positive, negative = [], []
    for review, result in zip(reviews, results):
        label = result["label"].lower()
        confidence = result["score"]
        bucket = positive if label == "positive" else negative
        bucket.append((review, confidence))
        print(f"[{label.upper():8s} {confidence:.0%}]  {review}")

    total = len(reviews)
    pos_count, neg_count = len(positive), len(negative)

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Total reviews analyzed : {total}")
    print(f"Positive               : {pos_count}  ({pos_count/total:.1%})")
    print(f"Negative               : {neg_count}  ({neg_count/total:.1%})")

    if negative:
        print("\nLowest-confidence / most negative reviews to look into first:")
        for review, conf in sorted(negative, key=lambda x: -x[1])[:3]:
            print(f"  - ({conf:.0%} negative) {review}")


if __name__ == "__main__":
    main()
