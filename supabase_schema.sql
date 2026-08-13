-- Run this once in the Supabase SQL Editor for the Document Q&A app.

create table if not exists public.document_chats (
    id text primary key,
    name text not null,
    created_at timestamptz not null default now()
);

create table if not exists public.chat_messages (
    id bigint generated always as identity primary key,
    chat_id text not null references public.document_chats(id) on delete cascade,
    question text not null,
    answer text not null,
    sources jsonb not null default '[]'::jsonb,
    created_at timestamptz not null default now()
);

create index if not exists chat_messages_chat_id_created_at_idx
    on public.chat_messages (chat_id, created_at);

-- Persistent vectorless hierarchy for documents classified as structured.
-- It is deliberately separate from the Qdrant vector collection.
create table if not exists public.structured_document_indexes (
    chat_id text not null references public.document_chats(id) on delete cascade,
    file_path text not null,
    file_name text not null,
    structure_score integer not null,
    tree jsonb not null,
    created_at timestamptz not null default now(),
    primary key (chat_id, file_path)
);

create index if not exists structured_document_indexes_chat_id_idx
    on public.structured_document_indexes (chat_id);
