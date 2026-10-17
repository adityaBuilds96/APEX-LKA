# APEX-Vision Core v3.0 Edge Container Environment
# Target Architectures: linux/arm64 (NVIDIA Jetson Orin Nano) & linux/amd64 (x86_64)

FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LANG=C.UTF-8

# Install system utilities, CAN networking, and multimedia libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    can-utils \
    iproute2 \
    net-tools \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgstrtspserver-1.0-0 \
    gstreamer1.0-tools \
    v4l-utils \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/apex

# Copy requirements and install Python dependencies
COPY requirements_edge.txt .
RUN pip install --no-cache-dir -r requirements_edge.txt \
    && pip install --no-cache-dir cryptography pyyaml

# Copy repository source code
COPY . /workspace/apex

# Expose CAN network interface permissions
VOLUME ["/tmp"]

# Production Headless Entry Point
ENTRYPOINT ["python3", "deploy/jetson_run.py"]
CMD ["--config", "configs/profiles/passenger_car.yaml", "--can-channel", "can0"]
