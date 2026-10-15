import subprocess
import sys
import time
import os

python_exe = os.path.join(".venv", "Scripts", "python.exe")
uvicorn_exe = os.path.join(".venv", "Scripts", "uvicorn.exe")
streamlit_exe = os.path.join(".venv", "Scripts", "streamlit.exe")

print("=" * 65)
print("🚀 Starting NimbusDesk AI - RAG Enterprise Hub")
print("=" * 65)
print("1. Launching FastAPI Backend on http://127.0.0.1:8000 ...")
backend_proc = subprocess.Popen(
    [uvicorn_exe, "backend.app.main:app", "--host", "127.0.0.1", "--port", "8000"],
    cwd=os.getcwd()
)

time.sleep(3)

print("2. Launching Streamlit Frontend on http://localhost:8501 ...")
frontend_proc = subprocess.Popen(
    [streamlit_exe, "run", "frontend/app.py", "--server.port", "8501"],
    cwd=os.getcwd()
)

print("\n" + "=" * 65)
print("✅ Services are running!")
print("   • Frontend UI:     http://localhost:8501")
print("   • Backend API:     http://127.0.0.1:8000")
print("   • API Swagger Docs: http://127.0.0.1:8000/docs")
print("=" * 65)
print("Press Ctrl+C to stop all services.\n")

try:
    frontend_proc.wait()
except KeyboardInterrupt:
    print("\nStopping services...")
    backend_proc.terminate()
    frontend_proc.terminate()
