# Dockerfile for building the OCR image.


# The base image, an example deep learning VM.
# For a full list, see https://us-docker.pkg.dev/deeplearning-platform-release/gcr.io/
# For info, see https://cloud.google.com/deep-learning-vm/docs/images#supported-frameworks
FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/base-cu118.py310

# Configures settings for the image.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore
WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxrender1 \
    libxext6 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*
    
# Installs your dependencies.
RUN pip install -U pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && rm -rf /tmp/*


# Copies your source files.
COPY src .

# Starts your model server.
CMD uvicorn ocr_server:app --port 5003 --host 0.0.0.0
