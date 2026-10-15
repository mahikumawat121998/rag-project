import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from backend.app.rag.chain import query_rag_pipeline

tests = [
    ("1. Greeting", "Hi"),
    ("2. Easy single-fact", "How many leave days do I get in a year?"),
    ("3. Follow-up using 'those'", "Which of those can I carry forward?"),
    ("4. Number from table", "Who approves an expense of INR 12,000?"),
    ("5. Different document", "What is the API rate limit on the Growth plan?"),
    ("6. Not in documents", "Does NimbusDesk support Telegram?"),
    ("7. Two documents", "My laptop was stolen while I was working from abroad. What should I do?"),
]

chat_history = []

for label, q in tests:
    print("=" * 65)
    print(f"TEST: {label}")
    print(f"QUERY: {q}")
    print("-" * 65)
    ans, sources = query_rag_pipeline(q, chat_history=chat_history)
    print(f"ANSWER:\n{ans}")
    print(f"SOURCES ({len(sources)}): {[s.source for s in sources]}")
    chat_history.append({"role": "user", "content": q})
    chat_history.append({"role": "assistant", "content": ans})

print("\nFinished evaluation run.")
