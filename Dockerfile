FROM nvidia/cuda:12.3.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies including Python 3.11, Node.js and npm
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    build-essential \
    curl \
    wget \
    vim \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /WarpFactory

# Install CuPy for CUDA 12.x (matching system CUDA 12.3)
RUN pip install --no-cache-dir cupy-cuda12x

# Install Python packages
RUN pip install --no-cache-dir \
    numpy>=1.24.0 \
    scipy>=1.10.0 \
    matplotlib>=3.7.0 \
    jupyter>=1.0.0 \
    notebook>=6.5.0 \
    sympy>=1.12 \
    pytest>=7.4.0 \
    black>=23.0.0 \
    pylint>=2.17.0

# Install Claude Code globally
RUN npm install -g @anthropic-ai/claude-code

# Create a non-root user for safety
RUN useradd -m -s /bin/bash warpuser && \
    chown -R warpuser:warpuser /WarpFactory

# Create .claude.json with API key
# IMPORTANT: Replace YOUR-ANTHROPIC-API-KEY-HERE with your actual API key from
# https://console.anthropic.com/settings/keys before building
RUN echo '{\
  "changelogLastFetched": 1000000000000,\
  "primaryApiKey": "YOUR-ANTHROPIC-API-KEY-HERE",\
  "isQualifiedForDataSharing": false,\
  "hasCompletedOnboarding": true,\
  "lastOnboardingVersion": "0.2.107",\
  "maxSubscriptionNoticeCount": 0,\
  "hasAvailableMaxSubscription": false,\
  "lastReleaseNotesSeen": "0.2.107"\
}' > /home/warpuser/.claude.json && \
    chown warpuser:warpuser /home/warpuser/.claude.json && \
    chmod 600 /home/warpuser/.claude.json

USER warpuser

# Default command
CMD ["/bin/bash"]
