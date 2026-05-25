"""
confirm_write.py
两个职责：
  1. append_to_pending() — 把 LLM 建议追加到 pending_review.json
  2. confirm_all()       — 读 pending_review.json，把 status=confirmed 的写入 DB
"""

import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path

DB_PATH      = os.environ.get("MEMORY_DB", "/home/ye/.hermes/memory_store.db")
PENDING_PATH = os.environ.get("PENDING_PATH", "pending_review.json")


# ════════════════════════════════════════════════════════════════
# Part 1: 写 pending_review.json
# ════════════════════════════════════════════════════════════════

def _load_pending() -> dict:
    p = Path(PENDING_PATH)
    if not p.exists():
        return {
            "generated_at": datetime.now().isoformat(),
            "l0_count": 0,
            "L1_proposals": [],
            "L3_proposals": [],
            "review_stats": {"total": 0, "pending": 0, "confirmed": 0, "rejected": 0},
        }
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_pending(data: dict):
    # 更新统计
    l1 = data.get("L1_proposals", [])
    l3 = data.get("L3_proposals", [])
    all_items = l1 + l3
    data["review_stats"] = {
        "total":     len(all_items),
        "pending":   sum(1 for x in all_items if x.get("status") == "pending"),
        "confirmed": sum(1 for x in all_items if x.get("status") == "confirmed"),
        "rejected":  sum(1 for x in all_items if x.get("status") == "rejected"),
    }
    data["generated_at"] = datetime.now().isoformat()
    with open(PENDING_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def append_to_pending(
    proposal: dict,
    trigger_type: str = "seed_expand",
    l0_count: int = 0,
    layer: str = "L1",
):
    """把一条建议追加到 pending_review.json，带去重"""
    data = _load_pending()

    # ── 去重检查 ───────────────────────────────────────────────
    proposals = data.get("L1_proposals", []) if layer != "L3" else data.get("L3_proposals", [])

    # ① ID 精确去重
    if any(p.get("id") == proposal.get("id") for p in proposals):
        print(f"  [append] 跳过（id重复）: {proposal.get('id')}")
        return

    # ② 新事实 ID 去重：过滤掉已在 confirmed L1 里的 new_fact_ids，
    #    若过滤后 new_fact_ids 非空 → 保留；空 → 丢弃整条（无新内容）
    #    注意：related_old_fact_ids 的关联信息不因 new_fact_ids 重复而丢弃
    if layer != "L3":
        new_ids = set(proposal.get("new_fact_ids", []))
        if new_ids:
            conn = sqlite3.connect(DB_PATH)
            try:
                rows = conn.execute(
                    "SELECT new_fact_ids FROM L1_categorized WHERE status='confirmed'"
                ).fetchall()
                confirmed_ids: set[int] = set()
                for (json_str,) in rows:
                    confirmed_ids.update(json.loads(json_str))
                overlap = new_ids & confirmed_ids
                filtered_new_ids = list(new_ids - confirmed_ids)
                if filtered_new_ids:
                    # 有新的内容（去重后），更新 proposal 中的 new_fact_ids
                    skipped = len(new_ids) - len(filtered_new_ids)
                    print(f"  [append] new_fact_ids 过滤重复 {skipped} 条，过渡到: {filtered_new_ids}")
                    proposal["new_fact_ids"] = filtered_new_ids
                else:
                    # 没有任何新内容，整条丢弃
                    print(f"  [append] 跳过（new_fact_ids 全部已确认）: {proposal.get('id')}  overlap={overlap}")
                    return
            finally:
                conn.close()

    # ── 正常追加 ─────────────────────────────────────────────
    data["l0_count"] = l0_count

    if layer == "L3":
        data["L3_proposals"].append(proposal)
    else:
        data["L1_proposals"].append(proposal)

    _save_pending(data)


# ════════════════════════════════════════════════════════════════
# Part 2: 确认写入 DB
# ════════════════════════════════════════════════════════════════

def _write_l1(conn: sqlite3.Connection, p: dict):
    conn.execute("""
        INSERT OR IGNORE INTO L1_categorized
        (id, type, new_fact_ids, related_old_fact_ids, similarity_scores,
         recall_sources, reason, suggested_summary, category, confidence,
         status, source, confirmed_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,'confirmed',?,datetime('now'))
    """, (
        p["id"],
        p.get("type", "label_only"),
        json.dumps(p.get("new_fact_ids", [])),
        json.dumps(p.get("related_old_fact_ids", [])),
        json.dumps(p.get("similarity_scores", [])),
        json.dumps(p.get("recall_sources", [])),
        p.get("reason", ""),
        p.get("suggested_summary", ""),
        p.get("category", "general"),
        p.get("confidence", 0.0),
        p.get("source", "seed_expand"),
    ))
    # source_l0_ids 单独处理：LLM 合并结果经 JSON round-trip 保留
    if p.get("source_l0_ids"):
        conn.execute("""
            UPDATE L1_categorized
            SET source_l0_ids = ?
            WHERE id = ?
        """, (json.dumps(p["source_l0_ids"]), p["id"]))
    print(f"  [write] L1 写入: {p['id']} ({p.get('category','')})")


def _write_l2(conn: sqlite3.Connection, p: dict):
    """
    L2 去重逻辑：
    1. 如果 can_merge=False，不生成 L2
    2. 如果同名 chunk_name 已存在 → 合并（追加 L1_id）
    3. 如果不存在 → 新建 L2 记录
    """
    if not p.get("can_merge"):
        return

    chunk_name = p.get("chunk_name", "").strip()
    if not chunk_name:
        return

    # 查是否已有同名 chunk
    existing = conn.execute("""
        SELECT id, L1_ids FROM L2_chunks
        WHERE chunk_name = ? AND status = 'confirmed'
    """, (chunk_name,)).fetchone()

    if existing:
        # 合并：追加新 L1_id 到现有 L2
        existing_id, existing_l1_ids = existing
        l1_list = json.loads(existing_l1_ids)
        if p["id"] not in l1_list:
            l1_list.append(p["id"])
            conn.execute("""
                UPDATE L2_chunks
                SET L1_ids = ?
                WHERE id = ?
            """, (json.dumps(l1_list), existing_id))
            print(f"  [write] L2 合并: {existing_id} ← {p['id']} (累计 {len(l1_list)} 条 L1)")
    else:
        # 新建
        l2_id = "L2_" + p["id"]
        source_l0_ids = p.get("source_l0_ids", [])
        conn.execute("""
            INSERT OR IGNORE INTO L2_chunks
            (id, chunk_name, L1_ids, source_l0_ids, context, category, status, source, confirmed_at)
            VALUES (?,?,?,?,?,?,'confirmed',?,datetime('now'))
        """, (
            l2_id,
            chunk_name,
            json.dumps([p["id"]]),
            json.dumps(source_l0_ids),
            p.get("suggested_summary", ""),
            p.get("category", "general"),
            p.get("source", "seed_expand"),
        ))
        print(f"  [write] L2 新建: {l2_id} ({chunk_name})")


def _write_l3(conn: sqlite3.Connection, p: dict):
    conn.execute("""
        INSERT OR IGNORE INTO L3_persona
        (id, dimension, content, source_L1_ids, confidence, status, source, confirmed_at)
        VALUES (?,?,?,?,?,'confirmed',?,datetime('now'))
    """, (
        p["id"],
        p.get("dimension", ""),
        p.get("content", ""),
        json.dumps(p.get("source_L1_ids", [])),
        p.get("confidence", 0.0),
        p.get("source", "monthly"),
    ))
    print(f"  [write] L3 写入: {p['id']} ({p.get('dimension','')})")


def _l1_confirmed_count(db_path: str) -> int:
    """查询 confirmed L1 总数，调用方负责连接管理"""
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM L1_categorized WHERE status='confirmed'"
        ).fetchone()[0]
    finally:
        conn.close()


def confirm_all():
    """
    读 pending_review.json，把 status='confirmed' 的记录写入 DB
    写完后将对应条目 status 改为已处理（避免重复写）
    L1 confirmed 累计达到阈值（默认100）时自动触发 L3 生成
    """
    L1_TRIGGER = int(os.environ.get("L1_TRIGGER_THRESHOLD", "100"))

    data = _load_pending()
    conn = sqlite3.connect(DB_PATH)

    written = 0

    # ── L1 / L2 ───────────────────────────────────────────────
    for p in data["L1_proposals"]:
        if p.get("status") == "confirmed":
            _write_l1(conn, p)
            _write_l2(conn, p)
            p["status"] = "written"
            written += 1

    # ── L3 ────────────────────────────────────────────────────
    for p in data["L3_proposals"]:
        if p.get("status") == "confirmed":
            _write_l3(conn, p)
            p["status"] = "written"
            written += 1

    conn.commit()
    conn.close()
    _save_pending(data)

    # ── L3 自动触发检查 ─────────────────────────────────────────
    l1_count = _l1_confirmed_count(DB_PATH)
    print(f"[confirm_write] L1 confirmed 总数: {l1_count}/{L1_TRIGGER}")

    if l1_count >= L1_TRIGGER:
        print(f"[confirm_write] 🎯 L1 达到阈值 {L1_TRIGGER}，自动触发 L3 生成")
        try:
            from monthly_persona import run as run_l3
            run_l3()
        except Exception as e:
            print(f"[confirm_write] ⚠️ L3 生成失败: {e}")

    print(f"[confirm_write] ✅ 共写入 {written} 条记录")
    return written


def show_pending():
    """打印当前待审核摘要"""
    data = _load_pending()
    stats = data.get("review_stats", {})
    print(f"\n📋 pending_review.json 状态")
    print(f"   生成时间 : {data.get('generated_at','')}")
    print(f"   L0 数量  : {data.get('l0_count', 0)}")
    print(f"   总计     : {stats.get('total', 0)}")
    print(f"   待确认   : {stats.get('pending', 0)}")
    print(f"   已确认   : {stats.get('confirmed', 0)}")
    print(f"   已拒绝   : {stats.get('rejected', 0)}")

    pending_l1 = [p for p in data.get("L1_proposals", []) if p.get("status") == "pending"]
    pending_l3 = [p for p in data.get("L3_proposals", []) if p.get("status") == "pending"]

    if pending_l1:
        print(f"\n── L1/L2 建议（{len(pending_l1)} 条待确认）──")
        for p in pending_l1:
            tag = "🔗 可合并L2" if p.get("can_merge") else "🏷️  仅标签"
            print(f"  [{p['id']}] {tag} | {p.get('category','')} | 置信度={p.get('confidence',0):.2f}")
            print(f"    摘要: {p.get('suggested_summary','')}")
            print(f"    原因: {p.get('reason','')}")

    if pending_l3:
        print(f"\n── L3 Persona 建议（{len(pending_l3)} 条待确认）──")
        for p in pending_l3:
            print(f"  [{p['id']}] {p.get('dimension','')} | 置信度={p.get('confidence',0):.2f}")
            print(f"    内容: {p.get('content','')}")


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"

    if cmd == "show":
        show_pending()
    elif cmd == "confirm":
        confirm_all()
    else:
        print("用法: python confirm_write.py [show|confirm]")
