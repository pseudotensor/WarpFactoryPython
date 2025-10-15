# WarpFactory GPU-Enabled Dockerfile
# Base image with CUDA support
FROM nvidia/cuda:12.3.0-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /WarpFactory/warpfactory_py

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
# Install CuPy for CUDA 12.x
RUN pip install cupy-cuda12x

# Install other requirements
RUN pip install -r requirements.txt

# Copy the entire project
COPY . .

# Install WarpFactory in development mode
RUN pip install -e .

# Create a non-root user
RUN useradd -m -s /bin/bash warpuser && \
    chown -R warpuser:warpuser /WarpFactory

USER warpuser

# Set default command
CMD ["/bin/bash"]
