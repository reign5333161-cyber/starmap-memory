"""
starmap_cleanup.py
每周六跑：只重建 L2 FTS 索引
（L3 检查已由 seed_expand → confirm_all 实时触发，不再重复跑）
"""
import sys, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/home/ye/.hermes/hermes-agent")

DB_PATH = os.environ.get("MEMORY_DB", "/home/ye/.hermes/memory_store.db")

def main():
    import sqlite3
    conn = sqlite3.connect(DB_PATH)

    from plugins.memory.holographic.store import MemoryStore
    store = MemoryStore()
    store.rebuild_l2_chunks_fts()
    l2_count = conn.execute(
        "SELECT COUNT(*) FROM L2_chunks WHERE status='confirmed'"
    ).fetchone()[0]
    conn.close()
    print(f"✅ L2 FTS 重建完成（confirmed L2: {l2_count} 条）")


if __name__ == "__main__":
    main()
