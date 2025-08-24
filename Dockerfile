FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    tesseract-ocr \
    libtesseract-dev \
    libleptonica-dev \
    poppler-utils \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

WORKDIR /app

COPY ./requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --timeout 300 --retries 3 -r requirements.txt && \
    python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"

COPY ./app /app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]