import sys
import requests
import json

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

API_URL = "http://127.0.0.1:8000/api/chat"

test_cases = [
    {
        "id": 1,
        "title": "Greeting (checks app doesn't search for chit-chat)",
        "question": "Hi",
        "expected_check": "Should be a warm greeting, no policy hallucination or irrelevant document search"
    },
    {
        "id": 2,
        "title": "Easy single-fact question (basic retrieval)",
        "question": "How many leave days do I get in a year?",
        "expected_check": "Should state 26 total: 8 Casual, 6 Sick, and 12 Earned"
    },
    {
        "id": 3,
        "title": "Follow-up using 'those' (checks chat history context)",
        "question": "Which of those can I carry forward?",
        "expected_check": "Should state only Earned Leave, up to a 30-day balance"
    },
    {
        "id": 4,
        "title": "A number from a table (checks table parsing)",
        "question": "Who approves an expense of INR 12,000?",
        "expected_check": "Should state the department head (claims between INR 5,001 and 25,000)"
    },
    {
        "id": 5,
        "title": "Different document (checks topic isolation)",
        "question": "What is the API rate limit on the Growth plan?",
        "expected_check": "Should state 500 requests per minute"
    },
    {
        "id": 6,
        "title": "Something not in documents (hallucination refusal)",
        "question": "Does NimbusDesk support Telegram?",
        "expected_check": "Should state it doesn't have that information / not mentioned"
    },
    {
        "id": 7,
        "title": "Cross-document question (needs 2 documents)",
        "question": "My laptop was stolen while I was working from abroad. What should I do?",
        "expected_check": "Should combine 2-hour reporting rule to security with company VPN abroad rules"
    }
]

history = []

for tc in test_cases:
    print("=" * 70)
    print(f"CASE {tc['id']}: {tc['title']}")
    print(f"QUERY:    '{tc['question']}'")
    print(f"EXPECTED: {tc['expected_check']}")
    print("-" * 70)

    payload = {
        "question": tc["question"],
        "chat_history": history
    }

    res = requests.post(API_URL, json=payload, timeout=40)
    assert res.status_code == 200, f"Failed with status {res.status_code}: {res.text}"

    data = res.json()
    answer = data.get("answer", "")
    sources = [s.get("source") for s in data.get("sources", [])]

    print(f"ACTUAL ANSWER:\n{answer.strip()}")
    print(f"CITED SOURCES ({len(sources)}): {sources}")
    print()

    # Update history for multi-turn
    history.append({"role": "user", "content": tc["question"]})
    history.append({"role": "assistant", "content": answer})

print("=" * 70)
print("ALL 7 BENCHMARK TESTS COMPLETED SUCCESSFULLY AGAINST LIVE API!")
print("=" * 70)
