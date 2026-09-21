"""
Review Sentiment Analyzer — Web UI

A simple local web app: upload a CSV of reviews, see each one classified
as positive/negative, plus a summary you can act on.

Run with:
    streamlit run app.py

Then your browser opens automatically at http://localhost:8501
"""

import pandas as pd
import streamlit as st
from transformers import pipeline

MODEL_PATH = "."  # change if your model files are in a subfolder instead

st.set_page_config(page_title="Review Sentiment Analyzer", page_icon="🍽️", layout="centered")


@st.cache_resource
def load_classifier():
    return pipeline("sentiment-analysis", model=MODEL_PATH, tokenizer=MODEL_PATH)


st.title("🍽️ Resturant Review  Analyzer System")
st.write("Power By SMSAMI.")
st.write("Upload a CSV of customer reviews and get an instant positive/negative breakdown.")

input_mode = st.radio(
    "How do you want to give me reviews?",
    ["Paste reviews (easiest, no CSV needed)", "Upload a CSV file"],
    horizontal=True,
)

reviews = None

if input_mode == "Paste reviews (easiest, no CSV needed)":
    st.caption("Paste one review per line below. Commas, quotes, emojis, any punctuation — all fine, "
               "just don't hit Enter in the middle of a single review.")
    pasted = st.text_area(
        "Reviews (one per line)",
        height=220,
        placeholder="Chief Burger as awesome love this food, we want to visit again\n"
                     "Generally the food here is OK but my best there is the Molten lava\n"
                     "Best burgers pizza must try n keep trying chicken steak burger, pizza",
    )
    if pasted.strip():
        reviews = [line.strip() for line in pasted.split("\n") if line.strip()]
        st.success(f"Loaded {len(reviews)} reviews from the text box.")

else:
    with st.expander("CSV format required"):
        st.write("One column named **review**, one review per row. Extra columns are fine, they're ignored.")
        st.write("If a review contains a comma, wrap the whole review in double quotes — "
                 "or just use the paste option above and skip this entirely.")
        st.code("review\n\"The food was amazing\"\n\"Service was slow and rude\"")

    uploaded_file = st.file_uploader("Upload your reviews CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Couldn't read that file as a CSV: {e}")
            st.info("Tip: switch to 'Paste reviews' above instead — it skips CSV formatting entirely.")
            st.stop()

        if "review" not in df.columns:
            st.error("This CSV needs a column named 'review'. Columns found: " + ", ".join(df.columns))
            st.stop()

        reviews = df["review"].astype(str).tolist()
        st.success(f"Loaded {len(reviews)} reviews.")

if reviews:
    if st.button("Analyze reviews", type="primary"):
        with st.spinner("Loading model and classifying..."):
            classifier = load_classifier()
            results = classifier(reviews)

        df = pd.DataFrame({
            "review": reviews,
            "sentiment": [r["label"] for r in results],
            "confidence": [round(r["score"], 3) for r in results],
        })

        pos_count = (df["sentiment"].str.lower() == "positive").sum()
        neg_count = (df["sentiment"].str.lower() == "negative").sum()
        total = len(df)

        st.subheader("Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total reviews", total)
        col2.metric("Positive", f"{pos_count} ({pos_count/total:.0%})")
        col3.metric("Negative", f"{neg_count} ({neg_count/total:.0%})")

        def highlight_sentiment(row):
            color = "#d4f7d4" if row["sentiment"].lower() == "positive" else "#f7d4d4"
            return [f"background-color: {color}"] * len(row)

        st.subheader("All reviews")
        st.dataframe(df.style.apply(highlight_sentiment, axis=1), use_container_width=True)

        st.subheader("Reviews to look into first (most confident negatives)")
        worst = df[df["sentiment"].str.lower() == "negative"].sort_values("confidence", ascending=False).head(5)
        st.table(worst[["review", "confidence"]])

        csv_out = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download results as CSV", csv_out, "review_results.csv", "text/csv")
else:
    st.info("Upload a CSV to get started, or try the sample_reviews.csv from earlier.")
