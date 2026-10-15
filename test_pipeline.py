import os
from dotenv import load_dotenv

load_dotenv(".env")

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

print("--- 1. Testing Document Loader ---")
def load_document(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return PyPDFLoader(file_path).load()
    elif ext in [".docx", ".doc"]:
        return Docx2txtLoader(file_path).load()
    elif ext in [".txt", ".md"]:
        return TextLoader(file_path, encoding="utf-8").load()
    return []

docs = []
for folder in ["document", "data/sample_docs"]:
    if os.path.exists(folder):
        for f in os.listdir(folder):
            path = os.path.join(folder, f)
            if os.path.isfile(path) and f.endswith((".pdf", ".docx", ".txt")):
                loaded = load_document(path)
                docs.extend(loaded)
                print(f"  Loaded {f}: {len(loaded)} part(s)")

print(f"\nTotal loaded documents/pages: {len(docs)}")

print("\n--- 2. Testing Chunking ---")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
chunks = text_splitter.split_documents(docs)
print(f"Total chunks created: {len(chunks)}")
print(f"Sample chunk: {chunks[0].page_content[:120]}...\n")

print("--- 3. Testing ChromaDB Vector Store & Gemini Embeddings ---")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
persist_dir = "data/chroma_db"

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_dir,
    collection_name="gemini_company_policy"
)
print(f"ChromaDB indexed successfully! Vector count: {vectorstore._collection.count()}")

print("\n--- 4. Testing Similarity Search ---")
query = "How many days of annual leave are allowed?"
results = vectorstore.similarity_search_with_relevance_scores(query, k=2)
for i, (doc, score) in enumerate(results):
    print(f"  Match {i+1} (Score: {score:.3f}): {doc.page_content[:120]}...")

print("\n--- 5. Testing Full LCEL RAG Chain ---")
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
prompt = ChatPromptTemplate.from_template(
    "Answer using context strictly:\n{context}\n\nQuestion: {question}\nAnswer:"
)
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.1)

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

ans = rag_chain.invoke(query)
print(f"Answer: {ans}")
print("\n[SUCCESS] PIPELINE TEST COMPLETED SUCCESSFULLY!")
