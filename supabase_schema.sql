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
