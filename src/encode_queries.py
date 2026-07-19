import numpy as np
import pandas as pd
import pyarrow as pa
from sentence_transformers import SentenceTransformer


INPUT_FILE = "candidate_data/calibration.f"
OUTPUT_DIR = "results"
OUTPUT_PATH = f"{OUTPUT_DIR}/calibration_query_embeddings.parquet"
MODEL_NAME = "BAAI/bge-m3"
INSTRUCTION = "Represent this sentence for searching relevant passages: "


def main():
    df = pd.read_feather(INPUT_FILE)
    print(f"Loaded {len(df)} queries from {INPUT_FILE}")

    query_ids = df["query_id"].tolist()
    query_texts = df["query_text"].tolist()
    ground_truths = df["ground_truth"].tolist()

    prefixed = [INSTRUCTION + q for q in query_texts]

    print(f"Loading model {MODEL_NAME} on CPU...")
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    print("Model loaded successfully")

    print("Encoding queries...")
    embeddings = model.encode(
        prefixed,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32,
    )
    print(f"Encoded {len(embeddings)} queries, shape: {embeddings.shape}")

    table = pa.table({
        "query_id": pa.array(query_ids, type=pa.int64()),
        "query_text": pa.array(query_texts, type=pa.string()),
        "ground_truth": pa.array(ground_truths, type=pa.string()),
        "embedding": pa.array([emb.tobytes() for emb in embeddings], type=pa.binary()),
    })
    table.to_pandas().to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved query embeddings to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
