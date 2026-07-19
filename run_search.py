import subprocess
import sys


def run_step(description: str, *args: str) -> None:
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, *args], capture_output=False)
    if result.returncode != 0:
        print(f"ERROR: step failed with code {result.returncode}")
        sys.exit(result.returncode)
    print(f"OK: {description}\n")


def main():
    run_step("Encode calibration queries", "src/encode_queries.py")
    run_step("Validate titles", "src/search.py", "--method", "dense_title", "--articles-path", "data/title_embeddings.parquet")
    run_step("Validate bodies", "src/search.py", "--method", "dense_body", "--articles-path", "data/article_body_embeddings.parquet")
    run_step("Rewrite and encode queries", "src/rewrite_queries.py")
    run_step("Validate rewritten → titles", "src/search.py", "--method", "dense_rewritten_title", "--articles-path", "data/title_embeddings.parquet", "--calib-path", "data/calibration_query_embeddings_rewritten.parquet")
    run_step("Validate rewritten → bodies", "src/search.py", "--method", "dense_rewritten_body", "--articles-path", "data/article_body_embeddings.parquet", "--calib-path", "data/calibration_query_embeddings_rewritten.parquet")

    print(f"\n{'='*60}")
    print("Search complete")
    print("  - data/calibration_query_embeddings.parquet")
    print("  - results/runs/dense_title.txt")
    print("  - results/runs/dense_body.txt")
    print("  - data/calibration_query_embeddings_rewritten.parquet")
    print("  - results/runs/dense_rewritten_title.txt")
    print("  - results/runs/dense_rewritten_body.txt")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
