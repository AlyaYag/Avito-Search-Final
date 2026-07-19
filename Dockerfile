FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch \
    sentence-transformers \
    transformers \
    pandas \
    pyarrow \
    numpy \
    scikit-learn \
    beautifulsoup4 \
    html2text

COPY src/ src/
COPY candidate_data/ candidate_data/
COPY run_pipeline.py run_search.py ./

RUN mkdir -p data

CMD ["python", "run_pipeline.py"]
