FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV CUDA_HOME=/usr/local/cuda-11.8/

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone GroundingDINO repo
RUN git clone https://github.com/IDEA-Research/GroundingDINO.git

# pip install torch==1.12.1+cu118 torchvision==0.13.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118 && \
# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116 && \
    pip install networkx==2.6.3

# Install GroundingDINO requirements
RUN cd GroundingDINO && \
    pip install -r requirements.txt && \
    pip install -e .

# Download pre-trained weights
RUN mkdir -p /app/weights && \
    wget -P /app/weights https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# Additional fixes:
# 1. Make "python" available as python3.
RUN ln -s $(which python3) /usr/bin/python
# 2. Pin numpy to an earlier version to avoid compatibility issues.
RUN pip install numpy==1.25.0

RUN pip install gradio==4.44.1

# Environment variables
ENV PYTHONPATH=/app/GroundingDINO
ENV CUDA_VISIBLE_DEVICES=0

# Copy the single_image.py file to the container
COPY demo/multiple_image.py /app/GroundingDINO/demo/multiple_image.py

# Copy the room.jpg file to the container
COPY demo/animals.jpg /app/GroundingDINO/demo/animals.jpg

# Install the package in development mode
# RUN cd GroundingDINO && python setup.py build develop --user

# Use JSON array format for CMD
CMD ["bash", "-c", "cd GroundingDINO && python3 demo/multiple_image.py"]
# CMD ["bash", "-c", "cd GroundingDINO && python3 demo/gradio_app.py"]
# CMD ["bash"]
