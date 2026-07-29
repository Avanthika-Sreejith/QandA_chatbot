FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=600 \
    PIP_RETRIES=5

COPY requirements.txt .
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc git \
    && python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --prefer-binary --retries 5 --timeout 120 -r requirements.txt \
    && apt-get purge -y --auto-remove gcc git \
    && rm -rf /var/lib/apt/lists/*

COPY . .
RUN mkdir -p /app/work/uploads && chmod -R 777 /app/work

EXPOSE 7860
CMD ["sh", "-c", "streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port ${PORT:-7860} --server.headless true --server.fileWatcherType none"]
