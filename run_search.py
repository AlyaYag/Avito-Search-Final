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
    run_step("Search (validate + answer.csv)", "src/search.py")

    print(f"\n{'='*60}")
    print("Search complete")
    print("  - results/calibration_query_embeddings.parquet")
    print("  - results/runs/dense_title.txt")
    print("  - results/answer.csv")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
