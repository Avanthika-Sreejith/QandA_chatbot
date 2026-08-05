---
title: Document Q&A Chatbot
emoji: 📄
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
suggested_hardware: t4-small
---

# AI-Powered Document Q&A Chatbot

An internship project for asking grounded questions across uploaded PDF, DOCX, XLSX, and TXT files. Version 1 excludes scanned and image-only documents.

## Day 1 setup

### Prerequisites

Install:

1. [Python 3.11+](https://www.python.org/downloads/) — select **Add Python to PATH** during installation.
2. [Docker Desktop](https://www.docker.com/products/docker-desktop/) — start it and wait for its status to show that Docker is running.

Close and reopen PowerShell, then verify:

```powershell
python --version
docker --version
docker compose version
```

### Create the Python environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Start Qdrant

```powershell
docker compose up -d
```

Qdrant is then available at `http://localhost:6333` and its dashboard is at `http://localhost:6333/dashboard`.

### Verify Qdrant

```powershell
python -m app.check_qdrant
```

Expected output: `Qdrant is reachable`.

## Day 2: test a document parser

With the virtual environment activated, run this for any text-based sample file:

```powershell
python -m app.test_parser "C:\path\to\your\sample.pdf"
```

The command prints extracted text previews and citation metadata. PDFs preserve page number; DOCX preserves headings; Excel preserves sheet and row range; TXT preserves line range.

## Day 3: test chunking

Chunk a supported document into overlapping retrieval-sized pieces:

```powershell
python -m app.test_chunker "C:\path\to\your\sample.pdf"
```

Every chunk retains the original source metadata, plus its chunk number and total number of chunks. The project uses Chonkie's `RecursiveChunker`, which prefers paragraph and sentence boundaries before falling back to word/character splits, so chunks preserve more complete ideas than fixed character windows.

## Day 4: create the hybrid-search collection

With Docker Desktop running and the virtual environment activated:

```powershell
python -m app.create_collection
```

This creates `document_chunks` with a 1024-dimension dense vector field for `Qwen/Qwen3-Embedding-0.6B` and a sparse BM25/IDF field. It never deletes an existing collection.

## Day 5: index documents

With Docker Desktop running and the short virtual environment activated, index a supported text-based file:

```powershell
python -m app.ingest "C:\path\to\your\sample.pdf"
```

To index every supported file in a folder and its subfolders:

```powershell
python -m app.ingest "C:\path\to\your\document-folder"
```

The first run downloads Qwen3 from Hugging Face and the BM25 model, so it can take several minutes. Each chunk is stored with its Qwen3 dense vector, BM25 sparse vector, original text, and source metadata.

Stop Qdrant when required:

```powershell
docker compose down
```

## Streamlit upload screen

Start Qdrant first, activate the virtual environment, then run:

```powershell
streamlit run app/streamlit_app.py
```

The screen accepts multiple PDF, DOCX, XLSX, XLSM, and TXT files at once. It also accepts a local folder path and indexes matching files recursively. A browser cannot natively upload an arbitrary folder to Streamlit, so the folder path option works when Streamlit is running on the same computer as the folder. For a deployed app, users can instead upload a ZIP archive: supported documents inside the archive (including nested folders) are extracted and indexed together.

### Hosting: preload embedding models

For a deployed app, set `PRELOAD_EMBEDDING_MODELS=true` in the hosting provider's environment variables. The server will load the embedding models during startup, before it accepts indexing requests. This keeps the first document upload from paying the model-loading delay.

## Planned structure

```
app/
  parsers/       # PDF, DOCX, XLSX and TXT extraction
  ingestion/     # chunking and indexing
  retrieval/     # hybrid Qdrant search
  generation/    # answer generation and citations
  check_qdrant.py
```

## Scope

Included: single files/folders, PDF, DOCX, XLSX, TXT, natural-language Q&A, and source references.

Excluded from V1: OCR, image-based files, and scanned documents.
