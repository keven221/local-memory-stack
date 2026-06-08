"""兼容入口：根目录 server.py 只转发到包内唯一 FastAPI 服务。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from local_memory_stack.server import app  # noqa: E402


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8900)
