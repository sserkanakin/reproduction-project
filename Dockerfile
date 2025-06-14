# FINAL VERSION: Use the most stable and common base image: Ubuntu 22.04 with CUDA 12.1
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.10, pip, and Git using apt-get (works on Ubuntu/Debian)
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set the Hugging Face cache directory inside the image
ENV HF_HOME=/huggingface_cache

# Copy the requirements file into the image
COPY requirements.txt /tmp/requirements.txt

# Install the Python dependencies
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# --- Pre-download the model to cache it within the image layer ---
# Copy and run the download script
COPY eval-pipeline/download_model.py /tmp/download_model.py
RUN python3 /tmp/download_model.py

# Set the working directory for when we run the container
WORKDIR /app

# By default, when the container runs, it will start a bash shell
CMD ["bash"]