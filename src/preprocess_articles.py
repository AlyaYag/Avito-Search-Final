import pandas as pd

from preprocess_html import preprocess_dataframe
from html2text_stage import convert_dataframe


INPUT_PATH = "candidate_data/articles.f"
OUTPUT_PATH = "data/articles_cleaned.f"


def main():
    df = pd.read_feather(INPUT_PATH)
    print(f"Loaded {len(df)} articles from {INPUT_PATH}")

    df = preprocess_dataframe(df, column="body")
    print("Stage 1: Avito HTML cleanup complete")

    df = convert_dataframe(df, column="body", output_column="body")
    print("Stage 2: html2text conversion complete")

    df.to_feather(OUTPUT_PATH)
    print(f"Saved preprocessed articles to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
