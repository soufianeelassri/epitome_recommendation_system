# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install system dependencies required by unstructured (for OCR, image processing, etc.)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    tesseract-ocr \
    libtesseract-dev \
    libleptonica-dev \
    poppler-utils \
    ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*  && apt-get clean

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY ./requirements.txt .

# Install any needed packages specified in requirements.txt with timeout and retry settings
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --timeout 300 --retries 3 -r requirements.txt

# Copy the rest of the application's code to the working directory
COPY ./app /app

# Command to run the application using uvicorn server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]