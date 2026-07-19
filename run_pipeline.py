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
    run_step("Preprocess HTML", "src/preprocess_html.py", "candidate_data/articles.f", "candidate_data/articles_cleaned.f")
    run_step("Convert HTML to text", "src/html2text_stage.py", "candidate_data/articles_cleaned.f", "candidate_data/articles_text.f")
    run_step("Encode article titles", "src/encode_articles.py")

    print(f"\n{'='*60}")
    print("Pipeline complete")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
