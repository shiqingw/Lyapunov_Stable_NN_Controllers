# Make sure you are using the 'devel' image
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

# Use bash shell
SHELL ["/bin/bash", "--login", "-c"]

# Set environment variables
ENV DEBIAN_FRONTEND=non-interactive \
    CUDA_HOME=/usr/local/cuda-12.1 \
    PATH="/usr/local/bin:${CUDA_HOME}/bin:${PATH}" \
    LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"\
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Install system packages, sorted and grouped for readability
RUN apt-get update -y && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    \
    # --- Build Tools & Libraries ---
    build-essential \
    cm-super \
    git \
    libboost-all-dev \
    libeigen3-dev \
    libglib2.0-dev \
    liblapack-dev \
    libopenblas-dev \
    libpoco-dev \
    \
    # --- System & CLI Utilities ---
    curl \
    gnupg-agent \
    htop \
    net-tools \
    software-properties-common \
    vim \
    \
    # --- Python ---
    python3-pip \
    python3-tk \
    \
    # --- Graphics/Display (X11/GL) ---
    dvipng \
    libegl1 \
    libgl1 \
    libglvnd0 \
    libglx0 \
    libx11-6 \
    libxext6 \
    \
    # --- TeX Live ---
    texlive-base \
    texlive-fonts-recommended \
    texlive-latex-extra \
    texlive-latex-recommended \
    \
    # --- Cleanup ---
    && texhash \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install dreal4
# This script will now work because gnupg-agent and software-properties-common are installed
RUN curl -fsSL https://raw.githubusercontent.com/dreal/dreal4/master/setup/ubuntu/22.04/install.sh | bash

# Install Python packages
# Use --no-cache-dir to avoid caching and save a layer
RUN python -m pip install --no-cache-dir \
    dreal \
    ipykernel \
    matplotlib \
    ninja \
    numpy==1.26.4 \
    scipy==1.12.0 \
    torchinfo \
    huggingface_hub \
    opencv-python

# Install Python packages required by this repository (neural_lyapunov_training)
# that are not already provided above. See requirements.txt.
#   - control:    imported by neural_lyapunov_training/dynamical_system.py (needed
#                 by both training and verification, since models.py imports it).
#   - hydra-core: configuration management for all training scripts (pulls in omegaconf).
#   - wandb:      imported unconditionally by the training scripts (even when disabled).
#   - tqdm:       used by neural_lyapunov_training/try_verifier.py.
RUN python -m pip install --no-cache-dir \
    control==0.9.4 \
    hydra-core==1.3.2 \
    wandb==0.16.0 \
    tqdm==4.66.1

# Install auto_LiRPA and alpha-beta-CROWN
RUN git clone --recursive https://github.com/Verified-Intelligence/alpha-beta-CROWN.git \
    && cd alpha-beta-CROWN \
    && git checkout 6b8bbcfac1c01da1cabd240a87e4dce1a65f5a2b \
    && git submodule update --init --recursive \
    && cd auto_LiRPA \
    && python -m pip install -e . \
    && cd .. \
    && cd complete_verifier \
    && python -m pip install -r requirements.txt
ENV PYTHONPATH="${PYTHONPATH}:/workspace/alpha-beta-CROWN:/workspace/alpha-beta-CROWN/complete_verifier:/workspace/alpha-beta-CROWN/auto_LiRPA"

# Clean up
RUN apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf ~/.cache/pip

# Spin the container to keep it running
CMD ["tail", "-f", "/dev/null"]