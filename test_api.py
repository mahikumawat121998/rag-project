import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

print("=" * 60)
print("1. Testing GET /health")
print("=" * 60)
res = client.get("/health")
print("HTTP Status:", res.status_code)
print("Response:", res.json())
assert res.status_code == 200

print("\n" + "=" * 60)
print("2. Testing GET /api/chat/status")
print("=" * 60)
res_status = client.get("/api/chat/status")
print("HTTP Status:", res_status.status_code)
print("Response:", res_status.json())
assert res_status.status_code == 200

print("\n" + "=" * 60)
print("3. Testing POST /api/chat")
print("=" * 60)
query = "How many days of paid annual leave do full-time employees receive?"
res_chat = client.post("/api/chat", json={"question": query})
print("HTTP Status:", res_chat.status_code)
assert res_chat.status_code == 200

data = res_chat.json()
print(f"\nUser Question: {data['question']}")
print(f"Assistant Answer:\n{data['answer']}")
print(f"\nCited Sources ({len(data['sources'])}):")
for s in data["sources"]:
    page_str = f"Page {s['page']}" if s.get('page') else "N/A"
    print(f"  • Document: {s['source']} | {page_str}")
    print(f"    Snippet: {s['content_preview']}")

print("\n" + "=" * 60)
print("[SUCCESS] ALL FASTAPI ENDPOINTS ARE WORKING PERFECTLY!")
print("=" * 60)
