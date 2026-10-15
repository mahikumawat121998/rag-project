import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

print("=" * 65)
print("RUNNING END-TO-END SYSTEM VERIFICATION")
print("=" * 65)

# 1. Health
r1 = client.get("/health")
assert r1.status_code == 200
print("[PASS] 1. GET /health -> 200 OK")

# 2. Status
r2 = client.get("/api/chat/status")
assert r2.status_code == 200
data2 = r2.json()
print(f"[PASS] 2. GET /api/chat/status -> Total Vectors: {data2['total_vectors']}")

# 3. Document list
r3 = client.get("/api/documents")
assert r3.status_code == 200
data3 = r3.json()
print(f"[PASS] 3. GET /api/documents -> Indexed Files: {data3['total_documents']} ({data3['total_chunks']} chunks)")

# 4. Document upload
sample_doc = b"Remote Expense Appendix:\nInternet allowance is reimbursed up to $60 per month for home broadband."
files = {"file": ("home_internet_policy.txt", sample_doc, "text/plain")}
r4 = client.post("/api/documents/upload", files=files)
assert r4.status_code == 200
data4 = r4.json()
print(f"[PASS] 4. POST /api/documents/upload -> Ingested '{data4['filename']}' ({data4['chunks_added']} chunk, total: {data4['total_vectors']})")

# 5. Query chat
r5 = client.post("/api/chat", json={"question": "What is the internet reimbursement limit?"})
assert r5.status_code == 200
data5 = r5.json()
print(f"[PASS] 5. POST /api/chat -> Question: 'What is the internet reimbursement limit?'")
print(f"      Answer: {data5['answer'][:100]}...")
print(f"      Sources: {len(data5['sources'])} cited")

print("=" * 65)
print("[ALL TESTS PASSED] RAG BACKEND & UPLOAD API ARE 100% OPERATIONAL!")
print("=" * 65)
