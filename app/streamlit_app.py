"""Streamlit upload screen for indexing V1 document types."""

from __future__ import annotations

from pathlib import Path
from shutil import copyfileobj
import sys
from datetime import datetime, timezone, timedelta
from typing import Any
from uuid import uuid4
from zipfile import BadZipFile, ZipFile

import streamlit as st
from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue

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
    delete_chat,
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


def delete_chat_documents(document_chat_id: str) -> None:
    """Permanently remove all indexed chunks belonging to one document chat."""
    client = get_client()
    if not client.collection_exists(QDRANT_COLLECTION):
        return
    client.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=FilterSelector(
            filter=Filter(
                must=[
                    FieldCondition(
                        key="document_chat_id",
                        match=MatchValue(value=document_chat_id),
                    )
                ]
            )
        ),
        wait=True,
    )


UPLOAD_DIRECTORY = Path("work/uploads")
SUPPORTED_TYPES = sorted(extension.removeprefix(".") for extension in SUPPORTED_EXTENSIONS)


def save_upload(uploaded_file: Any) -> Path:
    """Persist an uploaded file so the parser and Qdrant payload retain a source path."""
    UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)
    safe_name = Path(uploaded_file.name).name
    destination = UPLOAD_DIRECTORY / f"{uuid4().hex}_{safe_name}"
    destination.write_bytes(uploaded_file.getbuffer())
    return destination


def format_citation(result: dict[str, Any]) -> str:
    """Build an exact source citation from a retrieved chunk.

    Examples:
      Project_Management.pdf — page 4, chunk 2 of 6
      Budget.xlsx — Sheet: Expenses, rows 12–30, chunk 1 of 2
      Notes.docx — Section: Risk Management, paragraph 18
    """
    payload = result.get("payload") or {}
    file_name = Path(result.get("source", "unknown")).name
    source_type = payload.get("source_type", "")
    details: list[str] = []

    if source_type == "pdf" and payload.get("page"):
        details.append(f"page {payload['page']}")
    if payload.get("section"):
        details.append(f"Section: {payload['section']}")
    if payload.get("sheet"):
        details.append(f"Sheet: {payload['sheet']}")
    if payload.get("paragraph"):
        details.append(f"paragraph {payload['paragraph']}")
    if payload.get("table"):
        details.append(f"Table {payload['table']}")
    if payload.get("table_row_start") is not None and payload.get("table_row_end") is not None:
        details.append(f"rows {payload['table_row_start']}–{payload['table_row_end']}")
    if payload.get("chunk_index") is not None and payload.get("chunk_count"):
        details.append(f"chunk {payload['chunk_index'] + 1} of {payload['chunk_count']}")

    suffix = ", ".join(details)
    return f"{file_name} — {suffix}" if suffix else file_name


MAX_ZIP_FILES = 1_000
MAX_ZIP_UNCOMPRESSED_BYTES = 500 * 1024 * 1024


def extract_zip_upload(uploaded_file: Any) -> list[Path]:
    """Safely extract supported documents from an uploaded ZIP archive.

    The archive hierarchy is retained so that nested folders remain visible in
    the indexed-file list. Files outside the supported document types are
    ignored.
    """
    UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)
    archive_root = UPLOAD_DIRECTORY / f"{uuid4().hex}_{Path(uploaded_file.name).stem}"
    archive_root.mkdir(parents=True, exist_ok=False)
    extracted: list[Path] = []
    total_size = 0

    try:
        with ZipFile(uploaded_file) as archive:
            members = [info for info in archive.infolist() if not info.is_dir()]
            if len(members) > MAX_ZIP_FILES:
                raise ValueError(f"ZIP files may contain at most {MAX_ZIP_FILES} files.")

            for info in members:
                member_path = Path(info.filename)
                # Block absolute paths and ../ traversal (Zip Slip).
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise ValueError("ZIP contains an unsafe file path.")
                if member_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue

                total_size += info.file_size
                if total_size > MAX_ZIP_UNCOMPRESSED_BYTES:
                    raise ValueError("Supported files in the ZIP exceed the 500 MB extraction limit.")

                destination = archive_root / member_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, destination.open("wb") as target:
                    copyfileobj(source, target)
                extracted.append(destination)
    except BadZipFile as error:
        raise ValueError("The selected file is not a valid ZIP archive.") from error

    return extracted


def new_chat_name() -> str:
    """Create a readable default title for a new, initially empty chat."""
    IST = timezone(timedelta(hours=5, minutes=30))
    return f"Document chat — {datetime.now(IST).strftime('%d %b %Y, %H:%M')}"


def has_default_chat_name(chat_name: str) -> bool:
    """Return whether a chat has not yet received a meaningful user title."""
    return chat_name.startswith("Document chat —")


def title_for_uploads(uploads: list[Any]) -> str:
    """Create a useful first title from the selected upload names."""
    first_name = Path(uploads[0].name).stem
    return first_name if len(uploads) == 1 else f"{first_name} + {len(uploads) - 1} more"


def main() -> None:
    st.title("Document Q&A — V1")
    st.write("Add documents to the knowledge base before asking questions.")

    if "active_document_chat_id" not in st.session_state:
        st.session_state.active_document_chat_id = uuid4().hex
        st.session_state.active_document_chat_name = new_chat_name()

    saved_chats = get_document_chats()
    active_chat_id = st.session_state.active_document_chat_id
    active_chat_name = st.session_state.active_document_chat_name
    chat_options = [
        (active_chat_id, active_chat_name),
        *((chat_id, chat_name) for chat_id, chat_name in saved_chats.items() if chat_id != active_chat_id),
    ]
    if st.session_state.get("document_chat_picker") not in chat_options:
        st.session_state.document_chat_picker = chat_options[0]

    def create_document_chat() -> None:
        """Start an empty chat and persist it to Supabase."""
        chat_id = uuid4().hex
        chat_name = (
            st.session_state.get("new_document_chat_name", "").strip() or new_chat_name()
        )
        st.session_state.active_document_chat_id = chat_id
        st.session_state.active_document_chat_name = chat_name
        st.session_state.document_chat_picker = (chat_id, chat_name)
        st.session_state.new_document_chat_name = ""
        st.session_state.last_indexed_info = None
        upsert_chat(chat_id, chat_name)
        get_document_chats.clear()

    with st.sidebar:
        st.header("Document chats")
        selected_chat = st.selectbox(
            "Current chat",
            options=chat_options,
            format_func=lambda chat: chat[1],
            help="Queries search only the documents in the selected chat.",
            key="document_chat_picker",
        )
        selected_chat_id, selected_chat_name = selected_chat
        if selected_chat_id != active_chat_id:
            st.session_state.active_document_chat_id = selected_chat_id
            st.session_state.active_document_chat_name = selected_chat_name
            st.session_state.last_indexed_info = None
            st.rerun()

        st.text_input(
            "New chat name",
            placeholder="For example: Project management notes",
            key="new_document_chat_name",
        )
        st.button("New document chat", use_container_width=True, on_click=create_document_chat)

        renamed_chat = st.text_input(
            "Rename current chat",
            value=active_chat_name,
            key=f"rename_chat_{active_chat_id}",
        )
        if st.button("Save chat name", use_container_width=True):
            new_name = renamed_chat.strip()
            if new_name:
                st.session_state.active_document_chat_name = new_name
                upsert_chat(active_chat_id, new_name)
                get_document_chats.clear()
                st.rerun()

        with st.expander("Delete current chat"):
            st.warning("This permanently removes this chat, its Q&A history, and all files indexed in it.")
            confirmed = st.checkbox(
                "I understand this cannot be undone",
                key=f"confirm_delete_chat_{active_chat_id}",
            )
            if st.button(
                "Delete this chat",
                type="primary",
                disabled=not confirmed,
                use_container_width=True,
            ):
                try:
                    delete_chat_documents(active_chat_id)
                    delete_chat(active_chat_id)
                    st.session_state.chat_history_by_document_chat = {
                        chat_id: messages
                        for chat_id, messages in st.session_state.get("chat_history_by_document_chat", {}).items()
                        if chat_id != active_chat_id
                    }
                    new_id = uuid4().hex
                    new_name = new_chat_name()
                    st.session_state.active_document_chat_id = new_id
                    st.session_state.active_document_chat_name = new_name
                    st.session_state.last_indexed_info = None
                    get_document_chats.clear()
                    get_chat_files.clear()
                    st.rerun()
                except Exception as error:
                    st.error(f"Could not delete this chat: {error}")

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
    if "upload_widget_version" not in st.session_state:
        st.session_state.upload_widget_version = 0
    uploads = st.file_uploader(
        "Choose one or more documents",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
        help="Supported: PDF, DOCX, XLSX, XLSM, and TXT. Scanned/image-only files are not supported in V1.",
        key=f"upload_files_{st.session_state.upload_widget_version}",
    )
    if st.button("Index uploaded files", disabled=not uploads, use_container_width=True):
        try:
            from app.ingestion.indexing import ingest_files

            progress = st.progress(0, text="Preparing uploaded files…")

            def update_upload_progress(message: str, current: int, total: int) -> None:
                fraction = current / total if total else 0
                progress.progress(min(max(fraction, 0.0), 1.0), text=message)

            if has_default_chat_name(active_chat_name):
                active_chat_name = title_for_uploads(uploads)
                st.session_state.active_document_chat_name = active_chat_name

            saved_paths = [save_upload(uploaded) for uploaded in uploads]
            total_segments, total_chunks, skipped = ingest_files(
                saved_paths,
                progress_callback=update_upload_progress,
                document_chat_id=active_chat_id,
                document_chat_name=active_chat_name,
            )
            progress.progress(1.0, text="Indexing complete")
            upsert_chat(active_chat_id, active_chat_name)
            get_document_chats.clear()
            get_chat_files.clear()
            if skipped:
                notice = (
                    f"No text could be extracted from: {', '.join(skipped)}. "
                    "These may be empty, scanned, or image-only, so they were not indexed.\n\n"
                )
            else:
                notice = ""
            message = f"{notice}Indexed {len(uploads) - len(skipped)} file(s): {total_segments} segments and {total_chunks} chunks."
            st.session_state.last_indexed_info = message
            st.session_state.last_indexed_kind = "warning" if skipped else ("error" if not total_chunks else "info")
            st.session_state.upload_widget_version += 1
            st.rerun()
        except Exception as error:
            st.error(f"Indexing failed: {error}")

    st.caption("Or upload a ZIP to index all supported documents from a folder and its subfolders.")
    if "zip_upload_widget_version" not in st.session_state:
        st.session_state.zip_upload_widget_version = 0
    zip_upload = st.file_uploader(
        "Upload a folder as ZIP",
        type=["zip"],
        accept_multiple_files=False,
        help="The ZIP may contain PDF, DOCX, XLSX, XLSM, and TXT files in nested folders.",
        key=f"upload_zip_{st.session_state.zip_upload_widget_version}",
    )
    if st.button("Index ZIP folder", disabled=zip_upload is None, use_container_width=True):
        try:
            from app.ingestion.indexing import ingest_files

            progress = st.progress(0, text="Reading ZIP folder…")
            extracted_paths = extract_zip_upload(zip_upload)
            if not extracted_paths:
                raise ValueError("No supported documents were found in this ZIP.")

            def update_zip_progress(message: str, current: int, total: int) -> None:
                fraction = current / total if total else 0
                progress.progress(min(max(fraction, 0.0), 1.0), text=message)

            if has_default_chat_name(active_chat_name):
                active_chat_name = Path(zip_upload.name).stem or "Indexed folder"
                st.session_state.active_document_chat_name = active_chat_name

            total_segments, total_chunks, skipped = ingest_files(
                extracted_paths,
                progress_callback=update_zip_progress,
                document_chat_id=active_chat_id,
                document_chat_name=active_chat_name,
            )
            progress.progress(1.0, text="Indexing complete")
            upsert_chat(active_chat_id, active_chat_name)
            get_document_chats.clear()
            get_chat_files.clear()
            indexed_files = len(extracted_paths) - len(skipped)
            notice = ""
            if skipped:
                notice = f"No text could be extracted from: {', '.join(skipped)}.\n\n"
            st.session_state.last_indexed_info = (
                f"{notice}Indexed {indexed_files} file(s) from {zip_upload.name}: "
                f"{total_segments} segments and {total_chunks} chunks."
            )
            st.session_state.last_indexed_kind = "warning" if skipped else ("error" if not total_chunks else "info")
            st.session_state.zip_upload_widget_version += 1
            st.rerun()
        except Exception as error:
            st.error(f"ZIP indexing failed: {error}")

    if "last_indexed_info" not in st.session_state:
        st.session_state.last_indexed_info = None
    if "last_indexed_kind" not in st.session_state:
        st.session_state.last_indexed_kind = "info"
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
        kind = st.session_state.last_indexed_kind
        if kind == "warning":
            st.warning(st.session_state.last_indexed_info)
        elif kind == "error":
            st.error(st.session_state.last_indexed_info)
        else:
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
            # --- citations for only the chunks the answer used ---
            # Numbers match the "View retrieved chunks" list below.
            citation_lines: list[str] = []
            for idx, result in enumerate(results, start=1):
                if result.get("used", True):
                    citation_lines.append(f"[{idx}] {format_citation(result)}")
            if citation_lines:
                st.markdown("---")
                st.caption("**Sources**  \n" + "  \n".join(citation_lines))
            # --- retrieved-chunk evidence ---
            with st.expander("View retrieved chunks"):
                for idx, result in enumerate(results, start=1):
                    score = result.get("score") or 0.0
                    st.markdown(f"**[{idx}]** Similarity: {score:.2f}  \n{format_citation(result)}")
                    st.markdown((result.get("text") or "").strip())

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
                            "payload": payload,
                        }
                    )
                answer, used_numbers = generate_answer(query_text, results)
                valid_numbers = [n for n in used_numbers if 1 <= n <= len(results)]
                used_indices = sorted({n - 1 for n in valid_numbers}) if valid_numbers else list(range(len(results)))
                for index, result in enumerate(results):
                    result["used"] = index in used_indices
                append_message(active_chat_id, query_text, answer, results)
                chat_history.append({"question": query_text, "results": results, "answer": answer})
                st.rerun()
            except Exception as error:
                st.session_state.search_error = str(error)

    if st.session_state.search_error:
        st.error(f"Search failed: {st.session_state.search_error}")


if __name__ == "__main__":
    main()
