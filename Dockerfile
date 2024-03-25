# === BASE STAGE ===
FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
# Install essential packages and Python 3
RUN apt-get update && apt-get install -y \
    build-essential \
    autoconf \
    automake \
    libtool \
    tar \
    curl \
    git \
    tree \
    python3 \
    python3-pip \
    python3-dev

# Set work directory
WORKDIR /code

COPY . /code/
RUN pip3 install --no-cache-dir -r requirements.txt