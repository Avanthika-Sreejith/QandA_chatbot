"""Supabase database client and helpers for chat persistence."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()


@lru_cache(maxsize=1)
def get_supabase_client() -> Client:
    """Return a cached Supabase client."""
    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_KEY", "").strip()
    if not url or not key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_KEY must be set in .env to use database features."
        )
    return create_client(url, key)


# ---------------------------------------------------------------------------
# document_chats table
# ---------------------------------------------------------------------------

def upsert_chat(chat_id: str, chat_name: str) -> None:
    """Insert or update a document chat record."""
    client = get_supabase_client()
    client.table("document_chats").upsert(
        {"id": chat_id, "name": chat_name},
        on_conflict="id",
    ).execute()


def get_all_chats() -> dict[str, str]:
    """Return all document chats as {id: name}."""
    try:
        client = get_supabase_client()
        response = (
            client.table("document_chats")
            .select("id, name")
            .order("created_at", desc=False)
            .execute()
        )
        return {row["id"]: row["name"] for row in (response.data or [])}
    except Exception:
        return {}


def delete_chat(chat_id: str) -> None:
    """Delete a chat and its messages (via the database cascade)."""
    client = get_supabase_client()
    client.table("document_chats").delete().eq("id", chat_id).execute()


# ---------------------------------------------------------------------------
# chat_messages table
# ---------------------------------------------------------------------------

def get_messages(chat_id: str) -> list[dict[str, Any]]:
    """Return all Q&A messages for a chat, ordered oldest first."""
    try:
        client = get_supabase_client()
        response = (
            client.table("chat_messages")
            .select("question, answer, sources")
            .eq("chat_id", chat_id)
            .order("created_at", desc=False)
            .execute()
        )
        messages = []
        for row in response.data or []:
            messages.append(
                {
                    "question": row["question"],
                    "answer": row["answer"],
                    "results": row["sources"] if isinstance(row["sources"], list) else [],
                }
            )
        return messages
    except Exception:
        return []


def append_message(
    chat_id: str,
    question: str,
    answer: str,
    sources: list[dict[str, Any]],
) -> None:
    """Persist a single Q&A exchange to Supabase."""
    # Ensure the parent chat row exists before inserting a message
    client = get_supabase_client()
    client.table("chat_messages").insert(
        {
            "chat_id": chat_id,
            "question": question,
            "answer": answer,
            "sources": sources,
        }
    ).execute()


def clear_messages(chat_id: str) -> None:
    """Delete all messages for a given chat."""
    client = get_supabase_client()
    client.table("chat_messages").delete().eq("chat_id", chat_id).execute()
