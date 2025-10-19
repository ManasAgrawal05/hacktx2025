FROM python:3.12-slim

# 1) System packages required for building / runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    wget \
    git \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libglib2.0-0 \
    libheif-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2) Copy only dependency manifest (optional) - helps caching if you use requirements.txt
# COPY requirements.txt /app/    # <- use if you prefer requirements.txt

# 3) Upgrade pip / setuptools / wheel
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# 4) Pin numpy to <2 first (critical)
RUN pip install --no-cache-dir "numpy<2"

# 5) Install heavy compiled wheels (PyTorch). Choose CPU or CUDA wheel here.
#    CPU example (PyTorch official CPU wheels)
RUN pip install --no-cache-dir --upgrade \
    "torch" "torchvision" --index-url https://download.pytorch.org/whl/cpu

# 6) Install OpenCV (binary wheel)
RUN pip install --no-cache-dir opencv-python

# 7) Install facenet-pytorch which depends on torch
RUN pip install --no-cache-dir facenet-pytorch

# 8) Install pillow and HEIF support (pillow-heif) after system libs
RUN pip install --no-cache-dir "pillow>=11.1.0" pillow-heif

# 9) Install remaining Python packages (pure-Python or small wheels)
RUN pip install --no-cache-dir \
    flask \
    huggingface_hub \
    tqdm \
    pandas

# 10) Copy application code last so code changes only bust this layer
COPY server.py /app/server.py

# 11) Expose port and run
EXPOSE 5000
CMD ["python", "server.py"]