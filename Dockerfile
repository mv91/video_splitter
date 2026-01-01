FROM python:3.11-slim

# ffmpeg + runtime libs commonly needed by opencv-python-headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

RUN mkdir -p /data/in /data/out

ENV STORAGE_ACCOUNT=socialshopper
ENV CONTAINER=downloads
ENV INPUT_DIR=/data/in
ENV OUTPUT_DIR=/data/out
ENV THRESHOLD=27.0
ENV MIN_SCENE_LEN=15
ENV DEFAULT_IMAGES_PER_SCENE=3

ENV AZURE_CLIENT_ID=fdde68d7-9c90-41e2-b3a7-45511ae4b12b
ENV AZURE_CLIENT_SECRET=o~M8Q~x4d.bFBYN9IP1dtNzVJieoN-YfevtRtb4q
ENV AZURE_TENANT_ID=3574355d-5020-4cfa-8b79-ba2fdad16bf7

CMD ["python", "app.py"]
