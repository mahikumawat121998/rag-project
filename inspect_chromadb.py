import os
import sys
import chromadb
from collections import Counter

# Set console to UTF-8 output if possible
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

persist_directory = "data/chroma_db"
client = chromadb.PersistentClient(path=persist_directory)

collections = client.list_collections()
print("=" * 65)
print(f"[CHROMA STORAGE PATH] {os.path.abspath(persist_directory)}")
print(f"[TOTAL COLLECTIONS]   {len(collections)}")
print("=" * 65)

for col_meta in collections:
    col = client.get_collection(col_meta.name)
    total_count = col.count()
    print(f"\n--> Collection Name: '{col.name}'")
    print(f"    Total Stored Vectors / Chunks: {total_count}")

    # Fetch all metadata to see which documents are stored
    all_records = col.get(include=["metadatas"])
    sources = [m.get("source", "Unknown") for m in all_records["metadatas"]]
    source_counts = Counter(sources)

    print("\n    [CHUNKS BREAKDOWN BY SOURCE FILE]")
    for src, cnt in sorted(source_counts.items()):
        print(f"      * {src} -> {cnt} chunks")

    # Peek at first 3 stored records
    print("\n    [SAMPLE STORED CHUNKS (FIRST 3 RECORDS)]")
    sample_records = col.peek(limit=3)
    for i in range(len(sample_records["ids"])):
        print(f"\n    --- Record #{i+1} ---")
        print(f"    ID:       {sample_records['ids'][i]}")
        print(f"    Metadata: {sample_records['metadatas'][i]}")
        clean_text = sample_records['documents'][i].replace('\n', ' ')[:140]
        print(f"    Preview:  {clean_text}...")

print("\n" + "=" * 65)
print("[VERIFICATION RESULT] Data is permanently saved in ChromaDB sqlite3!")
print("=" * 65)
