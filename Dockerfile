# Dockerfile

ARG BASE_IMAGE=pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
FROM ${BASE_IMAGE}

WORKDIR /workspace

# Copy model + eval code
COPY llava-next/      ./llava-next/
COPY eval-pipeline/   ./eval-pipeline/

# Copy your requirements file
COPY requirements.txt .

# Install Python deps
RUN pip install --upgrade pip && \
    pip install -e llava-next && \
    pip install -r requirements.txt

CMD ["bash"]
