"""
01_batch_init.py
存量洗牌：遍历所有 is_promoted=0 的 L0 fact，批量触发 seed_expand
is_promoted=0 表示从未被分析过
"""

import sqlite3
import os
import time

from seed_expand import run as seed_expand_run

DB_PATH = os.environ.get("MEMORY_DB", "/home/ye/.hermes/memory_store.db")
BATCH_SIZE = 20   # 每批处理条数（控制 API 调用速率）


def get_unpromoted_facts(limit: int = 100) -> list[tuple[int, str]]:
    """
    读取尚未被分析过的 L0 facts
    返回 [(fact_id, content), ...]
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # 过滤条件：is_promoted = 0 表示未分析过
    rows = cur.execute("""
        SELECT fact_id, content
        FROM   facts
        WHERE  is_promoted = 0
        ORDER BY fact_id ASC
        LIMIT  ?
    """, (limit,)).fetchall()

    conn.close()
    return rows


def mark_promoted(fact_ids: list[int]):
    """标记为已分析（防止重复处理）"""
    if not fact_ids:
        return
    conn = sqlite3.connect(DB_PATH)
    placeholders = ",".join("?" * len(fact_ids))
    conn.execute(f"""
        UPDATE facts SET is_promoted = 1
        WHERE fact_id IN ({placeholders})
    """, fact_ids)
    conn.commit()
    conn.close()


def run():
    # 只查一次，缓存总数量和实际数据
    stats = get_unpromoted_facts(limit=999999)
    total = len(stats)
    if total == 0:
        print("[batch] 没有找到未分析的 L0 facts")
        return

    print(f"[batch] 开始存量洗牌，目标：{total} 条 L0 facts")
    print(f"[batch] 分批处理，每批 {BATCH_SIZE} 条")

    offset = 0
    processed = 0

    while True:
        batch = get_unpromoted_facts(limit=BATCH_SIZE)
        if not batch:
            break

        fact_ids = [f[0] for f in batch]
        contents = {f[0]: f[1] for f in batch}

        print(f"\n[b batch] 处理第 {offset+1}-{offset+len(batch)} 条...")
        for fact_id, content in batch:
            try:
                seed_expand_run(fact_id, content)
                time.sleep(0.3)   # 控制 API 调用频率
            except Exception as e:
                print(f"[batch] ⚠️  fact_id={fact_id} 处理失败: {e}")

        mark_promoted(fact_ids)
        processed += len(batch)
        offset += len(batch)
        print(f"[batch] ✅ 已处理 {processed} 条")

    print(f"\n[batch] 🎉 存量洗牌完成，共处理 {processed} 条")


if __name__ == "__main__":
    run()
