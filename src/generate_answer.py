import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from validate import load_embeddings


TEST_PATH = "candidate_data/test.f"
ARTICLES_PATH = "data/title_embeddings.parquet"
OUTPUT_PATH = "results/answer.csv"
MODEL_NAME = "BAAI/bge-m3"
INSTRUCTION = "Represent this sentence for searching relevant passages: "
TOP_K = 10
BATCH_SIZE = 64


def main():
    print(f"Loading model {MODEL_NAME} on CPU...")
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    print("Model loaded")

    test = pd.read_feather(TEST_PATH)
    print(f"Loaded {len(test)} test queries from {TEST_PATH}")

    prefixed = [INSTRUCTION + q for q in test["query_text"].tolist()]
    print("Encoding test queries...")
    query_emb = model.encode(
        prefixed,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=BATCH_SIZE,
    )
    print(f"Encoded {len(query_emb)} queries, shape: {query_emb.shape}")

    title_emb, title_df = load_embeddings(ARTICLES_PATH)
    article_ids = title_df["article_id"].values
    print(f"Loaded {len(article_ids)} article title embeddings, dim: {title_emb.shape[1]}")

    sim = query_emb @ title_emb.T
    print(f"Similarity matrix: {sim.shape}")

    answers = []
    for i in range(len(test)):
        scores = sim[i]
        top_indices = np.argsort(scores)[::-1][:TOP_K]
        top_ids = article_ids[top_indices]
        answers.append(" ".join(str(aid) for aid in top_ids))

    answer_df = pd.DataFrame({
        "query_id": test["query_id"].values,
        "answer": answers,
    })
    answer_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(answer_df)} answers to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
