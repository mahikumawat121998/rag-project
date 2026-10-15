import os
import sys
from dotenv import load_dotenv

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv(".env")

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
google_key = os.getenv("GOOGLE_API_KEY")

if not supabase_url or not supabase_key or "your-project-id" in supabase_url:
    print("=" * 65)
    print("⚠️  MISSING SUPABASE CREDENTIALS")
    print("Please set SUPABASE_URL and SUPABASE_SERVICE_KEY in your .env file:")
    print("  SUPABASE_URL=https://<your-project-id>.supabase.co")
    print("  SUPABASE_SERVICE_KEY=eyJhbGciOi...")
    print("=" * 65)
    sys.exit(1)

from supabase.client import create_client
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

print("=" * 65)
print("🚀 INGESTING COMPANY DOCUMENTS INTO SUPABASE PGVECTOR")
print("=" * 65)

# 1. Initialize Supabase Client
supabase_client = create_client(supabase_url, supabase_key)
print(f"✅ Connected to Supabase at: {supabase_url}")

# 2. Initialize Gemini Embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=google_key
)
print("✅ Initialized Google Gemini Embeddings (dim: 3072)")

# 3. Load Documents from document/ and data/sample_docs/
def load_document(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return PyPDFLoader(file_path).load()
    elif ext in [".docx", ".doc"]:
        return Docx2txtLoader(file_path).load()
    elif ext in [".txt", ".md"]:
        return TextLoader(file_path, encoding="utf-8").load()
    return []

folders = ["document", "data/sample_docs"]
docs = []

for folder in folders:
    if os.path.exists(folder):
        for f in os.listdir(folder):
            path = os.path.join(folder, f)
            if os.path.isfile(path) and f.endswith((".pdf", ".docx", ".txt", ".md")):
                loaded = load_document(path)
                # Normalize filename for metadata
                for doc in loaded:
                    doc.metadata["source"] = f
                docs.extend(loaded)
                print(f"  📄 Loaded {f}: {len(loaded)} part(s)")

print(f"\n📊 Total documents/pages: {len(docs)}")

# 4. Chunk Documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
chunks = text_splitter.split_documents(docs)
print(f"✂️  Created {len(chunks)} semantic chunks")

# 5. Index into Supabase via LangChain SupabaseVectorStore
print(f"\n⏳ Uploading and embedding {len(chunks)} chunks into Supabase table 'documents'...")

vector_store = SupabaseVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    client=supabase_client,
    table_name="documents",
    query_name="match_documents"
)

print(f"\n🎉 SUCCESS: All {len(chunks)} chunks are now stored in Supabase pgvector!")

# 6. Test retrieval
test_query = "How many leave days do I get in a year?"
print(f"\n🔍 Testing Supabase Retrieval for query: '{test_query}'")
results = vector_store.similarity_search(test_query, k=2)
for i, r in enumerate(results, 1):
    print(f"  Match #{i} [{r.metadata.get('source')}]: {r.page_content.strip()[:100]}...")

print("\n" + "=" * 65)
print("✅ SUPABASE READY FOR STREAMLIT COMMUNITY CLOUD DEPLOYMENT!")
print("=" * 65)
