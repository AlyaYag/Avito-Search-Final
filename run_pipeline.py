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
    run_step("Preprocess articles", "src/preprocess_articles.py")
    run_step("Encode queries", "src/encode_queries.py")
    run_step("Encode article titles", "src/encode_articles.py")
    run_step("Encode article bodies", "src/encode_article_bodies.py")
    run_step("Rewrite and encode queries", "src/rewrite_queries.py")

    print(f"\n{'='*60}")
    print("Pipeline complete")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
