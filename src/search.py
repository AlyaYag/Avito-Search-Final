import numpy as np
import pandas as pd

from src.rankers import dense_ranker
from src.validate import validate, load_embeddings


ARTICLES_PATH = "data/article_embeddings.parquet"
CALIB_PATH = "results/calibration_query_embeddings.parquet"
TEST_PATH = "candidate_data/test.f"
OUTPUT_PATH = "results/answer.csv"
TOP_K = 10
METHOD = "dense_title"


def main():
    article_emb, article_df = load_embeddings(ARTICLES_PATH)
    article_ids = article_df["article_id"].values
    print(f"Loaded {len(article_ids)} articles, embedding dim: {article_emb.shape[1]}")

    print("\n=== Validation on calibration data ===")
    validate(
        article_emb_path=ARTICLES_PATH,
        query_emb_path=CALIB_PATH,
        ranker=dense_ranker,
        method_name=METHOD,
        top_k=TOP_K,
    )

    print("\n=== Generating answer.csv for test queries ===")
    test = pd.read_feather(TEST_PATH)
    test_query_ids = test["query_id"].values
    test_query_texts = test["query_text"].tolist()

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-m3", device="cpu")
    prefixed = ["Represent this sentence for searching relevant passages: " + q for q in test_query_texts]
    test_emb = model.encode(prefixed, normalize_embeddings=True, show_progress_bar=True)

    test_rankings = dense_ranker(article_emb, article_ids, test_emb, top_k=TOP_K)

    answer = pd.DataFrame({
        "query_id": test_query_ids,
        "answer": [" ".join(str(aid) for aid in r) for r in test_rankings],
    })
    answer.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved answer.csv with {len(answer)} queries")


if __name__ == "__main__":
    main()
