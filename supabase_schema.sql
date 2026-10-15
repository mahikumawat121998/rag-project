-- ====================================================================
-- SUPABASE PGVECTOR SETUP FOR NIMBUSDESK RAG (Google Gemini Embeddings)
-- Embedding Dimension: 3072 (models/gemini-embedding-001)
-- ====================================================================

-- 1. Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Create the documents table
CREATE TABLE IF NOT EXISTS documents (
    id BIGSERIAL PRIMARY KEY,
    content TEXT NOT NULL,                -- Corresponds to Document.page_content
    metadata JSONB DEFAULT '{}'::jsonb,   -- Corresponds to Document.metadata (source, page, etc.)
    embedding VECTOR(3072),               -- 3072 dimensions for Google Gemini embeddings
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- 3. Create an HNSW index for ultra-fast cosine similarity search
CREATE INDEX IF NOT EXISTS documents_embedding_hnsw_idx 
ON documents 
USING hnsw (embedding vector_cosine_ops);

-- 4. Create the search function called by LangChain's SupabaseVectorStore
CREATE OR REPLACE FUNCTION match_documents (
    query_embedding VECTOR(3072),
    match_count INT DEFAULT 5,
    filter JSONB DEFAULT '{}'::jsonb
) 
RETURNS TABLE (
    id BIGINT,
    content TEXT,
    metadata JSONB,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
#variable_conflict use_column
BEGIN
    RETURN QUERY
    SELECT
        documents.id,
        documents.content,
        documents.metadata,
        1 - (documents.embedding <=> query_embedding) AS similarity
    FROM documents
    WHERE documents.metadata @> filter
    ORDER BY documents.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- 5. Enable Row Level Security (RLS) & Allow Read/Write via Service Key
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow public read access" 
ON documents 
FOR SELECT 
USING (true);

CREATE POLICY "Allow service role full access" 
ON documents 
FOR ALL 
USING (true);
