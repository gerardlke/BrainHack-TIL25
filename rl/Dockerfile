# Dockerfile for building the RL image.


# The base image. Using a regular Python base because RL agents don't typically
# require much GPU. This will speed up your evaluation.
FROM python:3.11-slim

# Configures settings for the image.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore
WORKDIR /workspace

# Installs your dependencies.
RUN pip install -U pip
COPY rl_torch_requirements.txt .
RUN pip install --no-cache-dir -r rl_torch_requirements.txt

# Copies your source files.
COPY src .
COPY model_folder /workspace/model_folder

# Starts your model server.
CMD uvicorn rl_server:app --port 5004 --host 0.0.0.0

# til submit runs docker tag, docker push, and gcloud ai models upload

# tag:
# asia-southeast1-docker.pkg.dev/til-ai-2025/mcdonalds-workers-repo-til-25/rl:0.0.1

# docker push

# gcloud ai models upload
# gcloud ai models upload --region