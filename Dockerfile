FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=600 \
    PIP_RETRIES=5 \
    PIP_REQUIRE_HASHES=0 \
    PIP_INDEX_URL=https://pypi.org/simple
ENV HF_HOME=/tmp/huggingface
ENV TRANSFORMERS_CACHE=/tmp/huggingface/transformers
ENV EMBEDDING_DEVICE=cuda

COPY requirements.txt .
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc git \
    && python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --prefer-binary --retries 5 --timeout 120 -r requirements.txt \
    && apt-get purge -y --auto-remove gcc git \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face GPU Spaces expose an NVIDIA GPU at runtime. Replace the CPU
# PyTorch dependency pulled by sentence-transformers with the CUDA build.
RUN python -m pip install --upgrade --force-reinstall \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu126

RUN mkdir -p /tmp/huggingface /tmp/huggingface/transformers \
    && chmod -R 777 /tmp/huggingface

COPY preload_models.py .
# Docker image builds do not have the Space GPU attached. Download the model
# weights on CPU now; the running GPU Space loads those same weights on CUDA.
RUN EMBEDDING_DEVICE=cpu python preload_models.py

COPY . .
RUN mkdir -p /app/work/uploads && chmod -R 777 /app/work

EXPOSE 7860
CMD ["sh", "-c", "streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port ${PORT:-7860} --server.headless true --server.fileWatcherType none"]
