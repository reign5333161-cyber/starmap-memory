"""
00_init_db.py
初始化 memory_store.db，新增 L1/L2/L3、向量缓存表、FTS5 全文索引
"""

import sqlite3
import os

DB_PATH = os.environ.get("MEMORY_DB", "/home/ye/.hermes/memory_store.db")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # ── 向量缓存表 ──────────────────────────────────────────────
    cur.execute("""
    CREATE TABLE IF NOT EXISTS facts_vectors (
        fact_id      INTEGER PRIMARY KEY,
        content_hash TEXT NOT NULL,
        vector       BLOB NOT NULL,          -- JSON 序列化的 float 列表
        dim          INTEGER NOT NULL,
        created_at   TEXT DEFAULT (datetime('now')),
        updated_at   TEXT DEFAULT (datetime('now'))
    )
    """)

    # ── L1 归类建议 ─────────────────────────────────────────────
    cur.execute("""
    CREATE TABLE IF NOT EXISTS L1_categorized (
        id              TEXT PRIMARY KEY,    -- p001, p002 ...
        type            TEXT NOT NULL,       -- merge_to_l2 | label_only
        new_fact_ids    TEXT,                -- JSON 数组 [623, 625]
        related_old_fact_ids TEXT,           -- JSON 数组 [12, 45, 88]
        similarity_scores    TEXT,           -- JSON 数组 [0.92, 0.87]
        recall_sources  TEXT,                -- JSON 数组，每条相关 fact 的召回来源（vector/bm25/both）
        reason          TEXT,
        suggested_summary TEXT,
        category        TEXT,
        confidence      REAL,
        status          TEXT DEFAULT 'pending',  -- pending|confirmed|rejected
        source          TEXT DEFAULT 'seed_expand',  -- seed_expand | monthly
        created_at      TEXT DEFAULT (datetime('now')),
        confirmed_at    TEXT
    )
    """)

    # ── L2 场景块 ───────────────────────────────────────────────
    cur.execute("""
    CREATE TABLE IF NOT EXISTS L2_chunks (
        id             TEXT PRIMARY KEY,
        chunk_name     TEXT NOT NULL,
        L1_ids         TEXT,                 -- JSON 数组，关联 L1 记录
        source_l0_ids  TEXT,                 -- JSON 数组，追踪链 L2→L0（来自 proposal 的同名字段）
        context        TEXT,                 -- 场景描述（Markdown 格式）
        category       TEXT,
        status         TEXT DEFAULT 'pending',
        source         TEXT DEFAULT 'seed_expand',
        created_at     TEXT DEFAULT (datetime('now')),
        confirmed_at   TEXT
    )
    """)

    # ── 确保现网 L2_chunks 表有 source_l0_ids 列（历史表升级兼容）──
    existing_cols = {
        row[1]
        for row in cur.execute("PRAGMA table_info(L2_chunks)").fetchall()
    }
    if "source_l0_ids" not in existing_cols:
        cur.execute("ALTER TABLE L2_chunks ADD COLUMN source_l0_ids TEXT")

    # ── L3 Persona ──────────────────────────────────────────────
    cur.execute("""
    CREATE TABLE IF NOT EXISTS L3_persona (
        id            TEXT PRIMARY KEY,
        dimension     TEXT NOT NULL,         -- 工具偏好/交易习惯/交互风格...
        content       TEXT NOT NULL,
        source_L1_ids TEXT,                  -- JSON 数组
        confidence    REAL,
        status        TEXT DEFAULT 'pending',
        source        TEXT DEFAULT 'monthly',
        created_at    TEXT DEFAULT (datetime('now')),
        confirmed_at  TEXT
    )
    """)

    # ── L3 触发水位线 ────────────────────────────────────────────
    # 记录每次月度 Persona 触发时已处理的 L1 id 集合，防止重复触发
    cur.execute("""
    CREATE TABLE IF NOT EXISTS l3_trigger_log (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        triggered_at   TEXT DEFAULT (datetime('now')),
        l1_id_set      TEXT NOT NULL,        -- JSON 数组，本次触发时全部 confirmed L1 的 id
        l1_count       INTEGER NOT NULL,
        trigger_status TEXT DEFAULT 'pending' -- pending | completed | failed
    )
    """)

    # ── 审核批次记录 ─────────────────────────────────────────────
    cur.execute("""
    CREATE TABLE IF NOT EXISTS review_batches (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        generated_at TEXT DEFAULT (datetime('now')),
        l0_count     INTEGER,
        trigger_type TEXT,                   -- seed_expand | monthly
        total        INTEGER DEFAULT 0,
        pending      INTEGER DEFAULT 0,
        confirmed    INTEGER DEFAULT 0,
        rejected     INTEGER DEFAULT 0,
        json_path    TEXT
    )
    """)

    # ── FTS5 全文检索虚拟表（BM25 混合召回）────────────────────────
    cur.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts
    USING fts5(
        content,                              -- 检索字段
        content_id UNINDEXED,                 -- 对应 facts.id，不参与索引
        tokenize = 'trigram case_sensitive 0' -- 支持中英文子串匹配，不区分大小写
    )
    """)

    conn.commit()
    conn.close()
    print(f"[init_db] ✅ 数据库表初始化完成: {DB_PATH}")


if __name__ == "__main__":
    init_db()
