import numpy as np
import pandas as pd
import pyarrow as pa
from sentence_transformers import SentenceTransformer


DATA_DIR = "candidate_data"
OUTPUT_DIR = "data"
ARTICLES_PATH = f"{DATA_DIR}/articles.f"
OUTPUT_PATH = f"{OUTPUT_DIR}/article_embeddings.parquet"
MODEL_NAME = "BAAI/bge-m3"


def main():
    articles = pd.read_feather(ARTICLES_PATH)
    print(f"Loaded {len(articles)} articles from {ARTICLES_PATH}")

    titles = articles["title"].tolist()
    article_ids = articles["article_id"].tolist()

    print(f"Loading model {MODEL_NAME} on CPU...")
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    print("Model loaded successfully")

    print("Encoding titles...")
    embeddings = model.encode(
        titles,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32,
    )
    print(f"Encoded {len(embeddings)} titles, shape: {embeddings.shape}")

    table = pa.table({
        "article_id": pa.array(article_ids, type=pa.int64()),
        "title": pa.array(titles, type=pa.string()),
        "embedding": pa.array([emb.tobytes() for emb in embeddings], type=pa.binary()),
    })
    table.to_pandas().to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved embeddings to {OUTPUT_PATH}")
    print(f"Table columns: {table.column_names}")


if __name__ == "__main__":
    main()
