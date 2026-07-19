import argparse

from rankers import dense_ranker
from validate import validate, load_embeddings


def main():
    parser = argparse.ArgumentParser(description="Validate article embeddings on calibration data")
    parser.add_argument("--articles-path", default="data/title_embeddings.parquet")
    parser.add_argument("--calib-path", default="data/calibration_query_embeddings.parquet")
    parser.add_argument("--method", default="dense_title")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    article_emb, article_df = load_embeddings(args.articles_path)
    article_ids = article_df["article_id"].values
    print(f"Loaded {len(article_ids)} articles, embedding dim: {article_emb.shape[1]}")

    print("\n=== Validation on calibration data ===")
    validate(
        article_emb_path=args.articles_path,
        query_emb_path=args.calib_path,
        ranker=dense_ranker,
        method_name=args.method,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
