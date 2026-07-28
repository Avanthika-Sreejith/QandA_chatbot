"""Streamlit upload screen for indexing V1 document types."""

from __future__ import annotations

from pathlib import Path
import sys
from concurrent.futures import Future
from datetime import datetime
from threading import Thread
from typing import Any
from uuid import uuid4

import streamlit as st
from qdrant_client.models import FieldCondition, Filter, MatchValue

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Document Q&A", page_icon="📄", layout="centered")

from app.parsers import SUPPORTED_EXTENSIONS
from app.retrieval.collection import QDRANT_COLLECTION, get_client
from app.database import (
    get_all_chats,
    upsert_chat,
    get_messages,
    append_message,
    clear_messages,
)


@st.cache_data(ttl=60, show_spinner=False)
def get_document_chats() -> dict[str, str]:
    """Return saved document chats from Supabase."""
    return get_all_chats()


@st.cache_data(ttl=60, show_spinner=False)
def get_chat_files(document_chat_id: str) -> list[str]:
    """Return every unique file stored in the selected document chat."""
    try:
        client = get_client()
        if not client.collection_exists(QDRANT_COLLECTION):
            return []

        files: set[str] = set()
        offset = None
        chat_filter = Filter(
            must=[
                FieldCondition(
                    key="document_chat_id",
                    match=MatchValue(value=document_chat_id),
                )
            ]
        )
        while True:
            points, offset = client.scroll(
                collection_name=QDRANT_COLLECTION,
                scroll_filter=chat_filter,
                limit=1000,
                offset=offset,
                with_payload=["file_name", "file_path"],
                with_vectors=False,
            )
            for point in points:
                payload = getattr(point, "payload", {}) or {}
                source_path = payload.get("file_path") or payload.get("file_name")
                if source_path:
                    files.add(str(source_path))
            if offset is None:
                break
        return sorted(files, key=str.casefold)
    except Exception:
        return []


@st.cache_resource(show_spinner=False)
def begin_model_preload() -> Future[None]:
    """Warm indexing models in the background after the app process starts."""
    completion: Future[None] = Future()

    def preload() -> None:
        try:
            from app.ingestion.indexing import get_embedding_models

            get_embedding_models()
        except Exception as error:
            completion.set_exception(error)
        else:
            completion.set_result(None)

    Thread(target=preload, name="embedding-model-preload", daemon=True).start()
    return completion


model_preload = begin_model_preload()


UPLOAD_DIRECTORY = Path("work/uploads")
SUPPORTED_TYPES = sorted(extension.removeprefix(".") for extension in SUPPORTED_EXTENSIONS)


def save_upload(uploaded_file: Any) -> Path:
    """Persist an uploaded file so the parser and Qdrant payload retain a source path."""
    UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)
    safe_name = Path(uploaded_file.name).name
    destination = UPLOAD_DIRECTORY / f"{uuid4().hex}_{safe_name}"
    destination.write_bytes(uploaded_file.getbuffer())
    return destination


def new_chat_name() -> str:
    """Create a readable default title for a new, initially empty chat."""
    return f"Document chat — {datetime.now().strftime('%d %b %Y, %H:%M')}"


def main() -> None:
    st.title("Document Q&A — V1")
    st.write("Add documents to the knowledge base before asking questions.")

    # Do not show upload controls until the indexing models are fully ready.
    # This prevents an upload from appearing to start while model warm-up is pending.
    if not model_preload.done():
        with st.spinner("Starting the document engine: loading Qwen embeddings and BM25…"):
            try:
                model_preload.result()
            except Exception as error:
                st.error(f"Document engine failed to start: {error}")
                return
    else:
        try:
            model_preload.result()
        except Exception as error:
            st.error(f"Document engine failed to start: {error}")
            return

    st.success("Document engine is ready.")

    if "active_document_chat_id" not in st.session_state:
        st.session_state.active_document_chat_id = uuid4().hex
        st.session_state.active_document_chat_name = new_chat_name()

    saved_chats = get_document_chats()
    active_chat_id = st.session_state.active_document_chat_id
    active_chat_name = st.session_state.active_document_chat_name
    chat_options = [active_chat_id, *(chat_id for chat_id in saved_chats if chat_id != active_chat_id)]
    if st.session_state.get("document_chat_picker") not in chat_options:
        st.session_state.document_chat_picker = active_chat_id

    def create_document_chat() -> None:
        """Start an empty chat and persist it to Supabase."""
        chat_id = uuid4().hex
        chat_name = (
            st.session_state.get("new_document_chat_name", "").strip() or new_chat_name()
        )
        st.session_state.active_document_chat_id = chat_id
        st.session_state.active_document_chat_name = chat_name
        st.session_state.document_chat_picker = chat_id
        st.session_state.new_document_chat_name = ""
        upsert_chat(chat_id, chat_name)
        get_document_chats.clear()

    with st.sidebar:
        st.header("Document chats")
        selected_chat_id = st.selectbox(
            "Current chat",
            options=chat_options,
            format_func=lambda chat_id: (
                active_chat_name if chat_id == active_chat_id else saved_chats[chat_id]
            ),
            help="Queries search only the documents in the selected chat.",
            key="document_chat_picker",
        )
        if selected_chat_id != active_chat_id:
            st.session_state.active_document_chat_id = selected_chat_id
            st.session_state.active_document_chat_name = saved_chats[selected_chat_id]
            st.rerun()

        st.text_input(
            "New chat name",
            placeholder="For example: Project management notes",
            key="new_document_chat_name",
        )
        st.button("New document chat", use_container_width=True, on_click=create_document_chat)

        st.caption("Previous chats and their documents stay available until you delete them from Qdrant.")

    active_chat_id = st.session_state.active_document_chat_id
    active_chat_name = st.session_state.active_document_chat_name
    st.subheader(active_chat_name)
    st.caption("Files added below belong to this chat only. Queries will not use documents from other chats.")

    chat_files = get_chat_files(active_chat_id)
    st.subheader("Files in this chat")
    if chat_files:
        st.success(f"{len(chat_files)} file(s) are already indexed for this chat.")
        with st.expander("Show indexed files"):
            for file_path in chat_files:
                st.write(f"• {file_path}")
    else:
        st.info("No files are indexed in this chat yet. Add files or a folder below.")

    st.subheader("Upload files")
    uploads = st.file_uploader(
        "Choose one or more documents",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
        help="Supported: PDF, DOCX, XLSX, XLSM, and TXT. Scanned/image-only files are not supported in V1.",
    )
    if st.button("Index uploaded files", disabled=not uploads, use_container_width=True):
        try:
            from app.ingestion.indexing import ingest_files

            saved_paths = [save_upload(uploaded) for uploaded in uploads]
            with st.spinner("Indexing your files…"):
                total_segments, total_chunks = ingest_files(
                    saved_paths,
                    document_chat_id=active_chat_id,
                    document_chat_name=active_chat_name,
                )
            upsert_chat(active_chat_id, active_chat_name)
            get_document_chats.clear()
            get_chat_files.clear()
            message = f"Indexed {len(uploads)} file(s): {total_segments} segments and {total_chunks} chunks."
            st.success(message)
            st.session_state.last_indexed_info = message
        except Exception as error:
            st.error(f"Indexing failed: {error}")

    st.divider()
    st.subheader("Index a local folder")
    folder_path = st.text_input(
        "Folder path",
        placeholder=r"C:\\Documents\\Project files",
        help="Indexes supported files in this folder and all subfolders. This path must be accessible to the computer running Streamlit.",
    )
    if st.button("Index folder", disabled=not folder_path.strip(), use_container_width=True):
        try:
            from app.ingestion.indexing import ingest_folder

            with st.spinner("Indexing folder…"):
                files, segments, chunks = ingest_folder(
                    folder_path.strip(),
                    document_chat_id=active_chat_id,
                    document_chat_name=active_chat_name,
                )
            upsert_chat(active_chat_id, active_chat_name)
            get_document_chats.clear()
            get_chat_files.clear()
            message = f"Indexed {files} file(s): {segments} segments and {chunks} chunks."
            st.success(message)
            st.session_state.last_indexed_info = message
        except Exception as error:
            st.error(f"Folder indexing failed: {error}")

    st.caption(
        "A web browser cannot reliably grant a Streamlit app access to an arbitrary folder on your computer. "
        "Use the folder path when Streamlit runs locally; otherwise upload files individually."
    )

    if "last_indexed_info" not in st.session_state:
        st.session_state.last_indexed_info = None
    if "query_text" not in st.session_state:
        st.session_state.query_text = ""
    if "search_results" not in st.session_state:
        st.session_state.search_results = []
    if "search_error" not in st.session_state:
        st.session_state.search_error = None
    if "search_done" not in st.session_state:
        st.session_state.search_done = False
    # Load chat history from Supabase (once per chat switch)
    cache_key = f"chat_loaded_{active_chat_id}"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = True
        st.session_state.setdefault("chat_history_by_document_chat", {})
        st.session_state.chat_history_by_document_chat[active_chat_id] = get_messages(active_chat_id)
    st.session_state.setdefault("chat_history_by_document_chat", {})
    chat_history = st.session_state.chat_history_by_document_chat.setdefault(active_chat_id, [])

    st.divider()
    st.subheader("Current chat indexing")
    if st.session_state.last_indexed_info:
        st.info(st.session_state.last_indexed_info)
    else:
        st.info("Index a file or folder in this chat to see its segment and chunk counts here.")

    st.divider()
    st.subheader("Query this document chat")
    st.caption(f"Answers are retrieved only from: {active_chat_name}")

    if st.button("Clear this chat's question history"):
        clear_messages(active_chat_id)
        st.session_state.chat_history_by_document_chat[active_chat_id] = []
        st.session_state.pop(f"chat_loaded_{active_chat_id}", None)
        st.rerun()

    for message in chat_history:
        with st.chat_message("user"):
            st.write(message["question"])
        with st.chat_message("assistant"):
            results = message.get("results", [])
            answer = message.get("answer", "")
            if not results:
                st.info("No relevant passages were found in this document chat.")
                continue
            # --- synthesised answer ---
            st.markdown(answer)
            # --- inline citations ---
            seen_sources: set[str] = set()
            citation_lines: list[str] = []
            citation_idx = 1
            for result in results:
                src = result["source"]
                if src not in seen_sources:
                    seen_sources.add(src)
                    citation_lines.append(f"[{citation_idx}] {Path(src).name}")
                    citation_idx += 1
            if citation_lines:
                st.markdown("---")
                st.caption("**Sources**  \n" + "  \n".join(citation_lines))

    query_text = st.chat_input("Ask a question about these documents")
    if query_text and query_text.strip():
        st.session_state.search_error = None
        with st.spinner("Searching and generating answer…"):
            try:
                # Import the embedding stack only when the user sends a question.
                from app.retrieval.search import search_documents
                from app.generation.generator import generate_answer

                hits = search_documents(query_text, document_chat_id=active_chat_id)
                results = []
                for hit in hits:
                    payload = hit.get("payload", {}) if isinstance(hit, dict) else getattr(hit, "payload", {}) or {}
                    score = hit.get("score") if isinstance(hit, dict) else getattr(hit, "score", None)
                    results.append(
                        {
                            "text": payload.get("text", ""),
                            "source": payload.get("file_path", payload.get("file_name", "unknown")),
                            "score": score,
                        }
                    )
                answer = generate_answer(query_text, results)
                append_message(active_chat_id, query_text, answer, results)
                chat_history.append({"question": query_text, "results": results, "answer": answer})
                st.rerun()
            except Exception as error:
                st.session_state.search_error = str(error)

    if st.session_state.search_error:
        st.error(f"Search failed: {st.session_state.search_error}")


if __name__ == "__main__":
    main()
