FROM nvidia/cuda:12.3.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies including Python 3.11, Node.js, npm, and MATLAB dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    build-essential \
    curl \
    wget \
    vim \
    libxt6 \
    libxrender1 \
    libxtst6 \
    libxi6 \
    libxrandr2 \
    libxcursor1 \
    libxdamage1 \
    libxcomposite1 \
    libxfixes3 \
    libfreetype6 \
    libfontconfig1 \
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
    chown -R warpuser:warpuser /WarpFactory && \
    mkdir -p /home/warpuser/.matlab/R2023b /home/warpuser/.claude && \
    chown -R warpuser:warpuser /home/warpuser/.matlab /home/warpuser/.claude

# Create a startup script to set up .claude.json from environment variable
RUN echo '#!/bin/bash\n\
if [ -n "$ANTHROPIC_API_KEY" ]; then\n\
  echo "{\n\
  \"changelogLastFetched\": 1000000000000,\n\
  \"primaryApiKey\": \"$ANTHROPIC_API_KEY\",\n\
  \"isQualifiedForDataSharing\": false,\n\
  \"hasCompletedOnboarding\": true,\n\
  \"lastOnboardingVersion\": \"0.2.107\",\n\
  \"maxSubscriptionNoticeCount\": 0,\n\
  \"hasAvailableMaxSubscription\": false,\n\
  \"lastReleaseNotesSeen\": \"0.2.107\"\n\
}" > /home/warpuser/.claude.json\n\
  chmod 600 /home/warpuser/.claude.json\n\
fi\n\
exec "$@"' > /usr/local/bin/docker-entrypoint.sh && \
    chmod +x /usr/local/bin/docker-entrypoint.sh && \
    chown -R warpuser:warpuser /WarpFactory

USER warpuser

# Set entrypoint to handle API key at runtime
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

# Default command
CMD ["/bin/bash"]
