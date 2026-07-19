import numpy as np
import pandas as pd
import pyarrow as pa
from sentence_transformers import SentenceTransformer


INPUT_PATH = "data/articles_cleaned.f"
OUTPUT_PATH = "data/article_body_embeddings.parquet"
MODEL_NAME = "BAAI/bge-m3"
INSTRUCTION = "Represent this sentence for searching relevant passages: "


def main():
    articles = pd.read_feather(INPUT_PATH)
    print(f"Loaded {len(articles)} articles from {INPUT_PATH}")

    article_ids = articles["article_id"].tolist()
    bodies = articles["body"].tolist()

    prefixed = [INSTRUCTION + b for b in bodies]

    print(f"Loading model {MODEL_NAME} on CPU...")
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    print("Model loaded successfully")

    print("Encoding article bodies...")
    embeddings = model.encode(
        prefixed,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=8,
    )
    print(f"Encoded {len(embeddings)} bodies, shape: {embeddings.shape}")

    table = pa.table({
        "article_id": pa.array(article_ids, type=pa.int64()),
        "embedding": pa.array([emb.tobytes() for emb in embeddings], type=pa.binary()),
    })
    table.to_pandas().to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved body embeddings to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
