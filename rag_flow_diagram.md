# NimbusDesk AI - Enterprise Policy RAG Architecture & Flow Diagrams

This document outlines the complete architectural design and data flow of the **NimbusDesk AI Enterprise Policy RAG** system.

---

## 1. High-Level System Architecture

The project supports two deployment patterns:
- **Local Microservices Architecture**: [Streamlit Frontend](file:///e:/RAG%20Project/frontend/app.py) communicates over REST API with [FastAPI Backend](file:///e:/RAG%20Project/backend/app/main.py) backed by local [ChromaDB](file:///e:/RAG%20Project/data/chroma_db).
- **Serverless Cloud Architecture**: Single [Streamlit App](file:///e:/RAG%20Project/streamlit_app.py) communicating with [Supabase pgvector](file:///e:/RAG%20Project/supabase_schema.sql) and [Google Gemini](https://ai.google.dev/).

```mermaid
graph TB
    subgraph Client["Client Layer (User Interface)"]
        UI["Streamlit Web UI<br/>(frontend/app.py / streamlit_app.py)"]
        QuickPrompts["Quick Prompt Buttons<br/>(Leave, Remote, Expense, Security)"]
        UploadWidget["Drag-and-Drop Document Ingestion Widget"]
        ChatHistory["Multi-turn Session History"]
    end

    subgraph Backend["API & Orchestration Layer"]
        FastAPI["FastAPI App (backend/app/main.py)<br/>Port: 8000"]
        ChatRouter["/api/chat (chat.py)"]
        DocRouter["/api/documents (documents.py)"]
        HealthRouter["/health & /docs (OpenAPI)"]
    end

    subgraph RAGCore["RAG Core Engine (LangChain + LCEL)"]
        Guardrails["Greeting & Chit-Chat Guardrail"]
        Reformulator["Contextual Query Reformulator"]
        Retriever["Vector Store Retriever (Top-K=5)"]
        PromptTemplate["Strict Grounding Prompt Template"]
        GeminiLLM["Google Gemini 3.5 Flash / Flash-Lite<br/>(temperature=0.1)"]
    end

    subgraph Storage["Vector Database & Storage Layer"]
        ChromaStore[("ChromaDB Persistent<br/>(data/chroma_db)")]
        SupabaseStore[("Supabase pgvector<br/>(PostgreSQL + HNSW index)")]
        DiskUploads["Persistent Document Storage<br/>(data/uploads)"]
    end

    subgraph Embeddings["Embeddings Model"]
        GeminiEmbed["models/gemini-embedding-001<br/>(768 dimensions)"]
    end

    UI -->|REST POST /api/chat| ChatRouter
    UI -->|REST POST /api/documents/upload| DocRouter
    ChatRouter --> Guardrails
    Guardrails -->|General Question| Reformulator
    Reformulator --> Retriever
    Retriever -->|Embed Query| GeminiEmbed
    GeminiEmbed -->|Dense Vector| ChromaStore
    GeminiEmbed -.->|Cloud Mode| SupabaseStore
    ChromaStore -->|Top-5 Semantic Chunks| PromptTemplate
    SupabaseStore -.->|Top-5 Chunks| PromptTemplate
    PromptTemplate --> GeminiLLM
    GeminiLLM -->|Grounded Response + Sources| ChatRouter
    ChatRouter -->|JSON Response| UI

    DocRouter --> DiskUploads
    DocRouter -->|Chunk & Embed| GeminiEmbed
    GeminiEmbed -->|Upsert Chunks| ChromaStore
```

---

## 2. Ingestion Pipeline Flow (Document Processing & Vectorization)

This flow runs during initial setup (via [test_pipeline.py](file:///e:/RAG%20Project/test_pipeline.py) or [ingest_to_supabase.py](file:///e:/RAG%20Project/ingest_to_supabase.py)) or at runtime when a user uploads a new policy document via the UI.

```mermaid
flowchart TD
    A["Raw Document (.pdf, .docx, .txt, .md)"] --> B{"File Type Validator"}
    
    B -->|PDF| C1["PyPDFLoader<br/>(Extracts pages & text)"]
    B -->|DOCX| C2["Docx2txtLoader<br/>(Extracts body text)"]
    B -->|TXT / MD| C3["TextLoader (UTF-8)"]
    B -->|Other| C4["HTTP 400 Bad Request / Reject"]

    C1 --> D["Metadata Tagging<br/>(source: filename, page: page_number)"]
    C2 --> D
    C3 --> D

    D --> E["RecursiveCharacterTextSplitter<br/>chunk_size: 500 characters<br/>chunk_overlap: 80 characters<br/>separators: [\\n\\n, \\n, ' ', '']"]

    E --> F["Normalized Text Chunks + Preserved Metadata"]

    F --> G["Google Gemini Embeddings<br/>Model: models/gemini-embedding-001<br/>Vector Dimensions: 768"]

    G --> H{"Target Vector Store Mode"}

    H -->|Local Mode| I["ChromaDB Collection<br/>'gemini_company_policy'<br/>data/chroma_db"]
    H -->|Cloud Mode| J["Supabase Table: 'documents'<br/>Index: HNSW (embedding vector_cosine_ops)"]

    I --> K["Ready for Vector Similarity Retrieval"]
    J --> K
```

### Key Ingestion Configurations
| Parameter | Setting | Rationale |
| :--- | :--- | :--- |
| **Embedding Model** | `models/gemini-embedding-001` | Generates 768-dimensional dense semantic vectors |
| **Chunk Size** | 500 characters | Retains concise, atomic policy rules & table rows |
| **Chunk Overlap** | 80 characters | Prevents sentence truncation across chunk boundaries |
| **Metadata Preserved** | `source`, `page`, `uploaded` | Enables exact source citation in UI responses |

---

## 3. End-to-End Query & Retrieval Flow (Conversational RAG Execution)

This sequence diagram depicts what happens when a user submits a prompt in the chat interface.

```mermaid
sequenceDiagram
    autonumber
    actor User as Employee / User
    participant UI as Streamlit UI
    participant API as FastAPI (/api/chat)
    participant Chain as RAG Chain (chain.py)
    participant LLM as Google Gemini (gemini-3.5-flash-lite)
    participant VS as Vector Store (ChromaDB / Supabase)

    User->>UI: Enters query or clicks quick prompt
    UI->>API: POST /api/chat { question, chat_history }
    API->>Chain: query_rag_pipeline(question, chat_history)

    Note over Chain: Step 1: Chit-Chat / Greeting Detection
    Chain->>Chain: is_greeting(question) regex check
    alt Is Greeting (e.g., "Hi", "Hello")
        Chain-->>API: Returns friendly greeting immediately (No search)
        API-->>UI: Displays greeting (0 vector search latency)
    else Is Policy Question
        Note over Chain: Step 2: Contextual Query Reformulation
        alt Chat History Exists (Multi-turn)
            Chain->>LLM: Reformulate question using recent 4 history turns
            LLM-->>Chain: Standalone search query (e.g. "Which of those..." -> "Which leave types can be carried forward?")
        else First Query
            Chain->>Chain: Use original question directly
        end

        Note over Chain: Step 3: Semantic Vector Retrieval
        Chain->>VS: similarity_search(standalone_query, k=5)
        VS-->>Chain: Top 5 Relevant Chunks + Metadata (file, page)

        Note over Chain: Step 4: Prompt Construction
        Chain->>Chain: format_docs(docs) + inject System Instructions

        Note over Chain: Step 5: Answer Generation
        Chain->>LLM: Invoke Gemini with Augmented Context
        LLM-->>Chain: Grounded Answer (Strict citations, figures, limits)

        Note over Chain: Step 6: Metadata Extraction
        Chain->>Chain: Parse filenames & page numbers into SourceDocument list
        Chain-->>API: Return (answer, sources)
        API-->>UI: HTTP 200 { question, answer, sources, success: true }
        UI->>User: Renders Markdown Answer + Expandable Cited Sources
    end
```

---

## 4. Multi-Turn Follow-Up Handling (Query Reformulation Detail)

A common challenge in RAG systems is pronoun ambiguity in conversational follow-ups. The diagram below illustrates how this project solves it in [chain.py](file:///e:/RAG%20Project/backend/app/rag/chain.py#L33-L60):

```mermaid
flowchart LR
    subgraph Input["Turn 1 & 2 Input"]
        T1["Turn 1 Question: 'How many leave days do I get in a year?'"]
        A1["Turn 1 Answer: 'You get 26 days: 8 Casual, 6 Sick, and 12 Earned leave.'"]
        T2["Turn 2 Question: 'Which of those can I carry forward?'"]
    end

    subgraph Reformulator["Reformulation Prompt (LangChain LCEL)"]
        RP["ChatPromptTemplate:<br/>'Given the chat history and latest user question,<br/>rephrase the question to be a standalone search query.<br/>Do NOT answer the question.'"]
    end

    subgraph LLM["Gemini LLM"]
        G["Gemini 3.5 Flash"]
    end

    subgraph Output["Refined Search Query"]
        SQ["'Which of the 26 annual leave types (Casual, Sick, Earned) can be carried forward?'"]
    end

    subgraph VectorSearch["ChromaDB / Supabase Search"]
        VS["Retriever returns exact section on 'Earned Leave carry-forward rules (up to 30 days)'"]
    end

    T1 & A1 & T2 --> RP
    RP --> G
    G --> SQ
    SQ --> VS
```

---

## 5. Dual Vector Store Architecture: ChromaDB vs. Supabase

The project is architected to switch between local development and cloud production:

```mermaid
graph TD
    subgraph Switch["Configuration Switch (streamlit_app.py)"]
        Check{"USE_SUPABASE = bool(SUPABASE_URL and SUPABASE_KEY)"}
    end

    subgraph LocalMode["Mode 1: Local / On-Premise"]
        FastAPI_App["FastAPI Backend (backend/app/main.py)"]
        ChromaInstance["ChromaDB Persistent (data/chroma_db)"]
        Collection["Collection: 'gemini_company_policy'"]
        LocalFiles["Local Filesystem uploads (data/uploads)"]
        
        FastAPI_App --> ChromaInstance
        ChromaInstance --> Collection
    end

    subgraph CloudMode["Mode 2: Cloud / Serverless"]
        StreamlitCloud["Streamlit Community Cloud"]
        SupabasePG["Supabase PostgreSQL Cloud"]
        PGVector["Extension: vector"]
        TableDocs["Table: documents (id, content, metadata, embedding VECTOR(768))"]
        HNSWIndex["Index: documents_embedding_hnsw_idx (vector_cosine_ops)"]
        RPCFunction["RPC: match_documents(query_embedding, match_count, filter)"]

        StreamlitCloud --> SupabasePG
        SupabasePG --> PGVector
        PGVector --> TableDocs
        TableDocs --> HNSWIndex
        TableDocs --> RPCFunction
    end

    Check -->|False / Default| LocalMode
    Check -->|True| CloudMode
```

---

## 6. Verification & Evaluation Framework

The project includes automated benchmark tests ([test_live_eval.py](file:///e:/RAG%20Project/test_live_eval.py) and [test_eval.py](file:///e:/RAG%20Project/test_eval.py)) verifying 7 real-world scenarios:

```mermaid
graph TD
    TestRunner["Evaluation Suite (test_live_eval.py)"]

    TestRunner --> TC1["Test 1: Greeting Filtering<br/>Ensures 'Hi' returns greeting without hallucination or doc search"]
    TestRunner --> TC2["Test 2: Single-Fact Retrieval<br/>'How many leave days?' -> verifies 26 total (8 Casual, 6 Sick, 12 Earned)"]
    TestRunner --> TC3["Test 3: Follow-Up Contextual Memory<br/>'Which of those can I carry forward?' -> verifies Earned Leave (30 days max)"]
    TestRunner --> TC4["Test 4: Tabular Numeric Extraction<br/>'Who approves expense of INR 12,000?' -> verifies Dept Head approval"]
    TestRunner --> TC5["Test 5: Topic Isolation<br/>'API rate limit on Growth plan?' -> queries product docs strictly"]
    TestRunner --> TC6["Test 6: Hallucination Refusal<br/>'Does NimbusDesk support Telegram?' -> verifies polite refusal"]
    TestRunner --> TC7["Test 7: Cross-Document Synthesis<br/>'Laptop stolen abroad?' -> combines IT security 2-hr rule + Abroad VPN policy"]
```

---

## 7. Directory & Component Mapping

| Component | File Path | Role |
| :--- | :--- | :--- |
| **API Application** | [main.py](file:///e:/RAG%20Project/backend/app/main.py) | FastAPI app instance, CORS middleware, lifespan setup |
| **Configuration** | [config.py](file:///e:/RAG%20Project/backend/app/config.py) | Pydantic Settings loading environment variables |
| **RAG Chain** | [chain.py](file:///e:/RAG%20Project/backend/app/rag/chain.py) | Greeting filter, query reformulation, LCEL QA chain |
| **Vector Store** | [vectorstore.py](file:///e:/RAG%20Project/backend/app/rag/vectorstore.py) | Singleton Chroma client and Gemini embeddings getter |
| **Document Ingestion** | [ingest.py](file:///e:/RAG%20Project/backend/app/rag/ingest.py) | Multi-format parser, recursive chunker, vector indexer |
| **Chat Endpoint** | [chat.py](file:///e:/RAG%20Project/backend/app/routers/chat.py) | `POST /api/chat`, `GET /api/chat/status` |
| **Document Endpoint**| [documents.py](file:///e:/RAG%20Project/backend/app/routers/documents.py) | `POST /api/documents/upload`, `GET /api/documents` |
| **Standalone Streamlit** | [streamlit_app.py](file:///e:/RAG%20Project/streamlit_app.py) | Dual-mode Streamlit frontend (Local + Cloud Supabase) |
| **Dedicated Frontend** | [app.py](file:///e:/RAG%20Project/frontend/app.py) | Client UI connecting to local FastAPI backend |
| **Supabase Setup** | [supabase_schema.sql](file:///e:/RAG%20Project/supabase_schema.sql) | DDL script for pgvector table, HNSW index & cosine match RPC |
| **Live Evaluation** | [test_live_eval.py](file:///e:/RAG%20Project/test_live_eval.py) | 7-stage automated RAG quality benchmark |
