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
COPY run_pipeline.py run_search.py ./

CMD ["python", "run_search.py"]
