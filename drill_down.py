"""
drill_down.py
追踪链下钻：从任意层 ID 向下追溯到 L0 原始事实

支持：
  drill_down(id)         → 返回结构化 dict，Agent 程序调用
  drill_down_report(id)  → 返回格式化字符串，可直接注入 prompt
  cli()                  → 命令行调用

用法示例：
  python3 drill_down.py l3001
  python3 drill_down.py L2_p001
  python3 drill_down.py p001
"""

import sqlite3
import json
import os
import sys
from typing import Any

DB_PATH = os.environ.get("MEMORY_DB", "/home/ye/.hermes/memory_store.db")


# ════════════════════════════════════════════════════════════════
# 底层查询
# ════════════════════════════════════════════════════════════════

def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _fetch_l3(l3_id: str) -> dict | None:
    conn = _conn()
    row = conn.execute(
        "SELECT * FROM L3_persona WHERE id = ?", (l3_id,)
    ).fetchone()
    conn.close()
    if not row:
        return None
    return dict(row)


def _fetch_l2(l2_id: str) -> dict | None:
    conn = _conn()
    row = conn.execute(
        "SELECT * FROM L2_chunks WHERE id = ?", (l2_id,)
    ).fetchone()
    conn.close()
    if not row:
        return None
    return dict(row)


def _fetch_l1(l1_id: str) -> dict | None:
    conn = _conn()
    row = conn.execute(
        "SELECT * FROM L1_categorized WHERE id = ?", (l1_id,)
    ).fetchone()
    conn.close()
    if not row:
        return None
    return dict(row)


def _fetch_l0_batch(fact_ids: list[int]) -> list[dict]:
    if not fact_ids:
        return []
    conn = _conn()
    ph   = ",".join("?" * len(fact_ids))
    rows = conn.execute(
        f"SELECT fact_id, content, created_at FROM facts WHERE fact_id IN ({ph})",
        fact_ids
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _parse_json_ids(raw: str | None) -> list:
    """安全解析 JSON 数组字段"""
    if not raw:
        return []
    try:
        val = json.loads(raw)
        return val if isinstance(val, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


# ════════════════════════════════════════════════════════════════
# 各层下钻逻辑
# ════════════════════════════════════════════════════════════════

def _resolve_l1_to_l0(l1_ids: list[str]) -> tuple[list[dict], list[int]]:
    """从 L1 列表 → 拿到所有 L0 fact_ids，再拉取原文"""
    all_fact_ids: list[int] = []
    l1_records:   list[dict] = []

    for l1_id in l1_ids:
        rec = _fetch_l1(l1_id)
        if not rec:
            continue
        l1_records.append(rec)
        # new_fact_ids + related_old_fact_ids 合并
        ids = (
            _parse_json_ids(rec.get("new_fact_ids")) +
            _parse_json_ids(rec.get("related_old_fact_ids"))
        )
        all_fact_ids.extend(int(i) for i in ids if str(i).isdigit())

    # 去重保序
    seen, unique = set(), []
    for fid in all_fact_ids:
        if fid not in seen:
            seen.add(fid)
            unique.append(fid)

    l0_records = _fetch_l0_batch(unique)
    return l1_records, l0_records


def _chain_from_l3(l3_id: str) -> dict:
    l3 = _fetch_l3(l3_id)
    if not l3:
        return {"error": f"L3 id={l3_id} 不存在"}

    # L3 → L1（source_L1_ids）
    l1_ids = _parse_json_ids(l3.get("source_L1_ids"))

    # 尝试通过 L1 找关联的 L2
    l2_records: list[dict] = []
    conn = _conn()
    for l1_id in l1_ids:
        rows = conn.execute(
            "SELECT * FROM L2_chunks WHERE L1_ids LIKE ?",
            (f'%"{l1_id}"%',)
        ).fetchall()
        for r in rows:
            rec = dict(r)
            if not any(x["id"] == rec["id"] for x in l2_records):
                l2_records.append(rec)
    conn.close()

    l1_records, l0_records = _resolve_l1_to_l0(l1_ids)

    return {
        "query_id":    l3_id,
        "query_layer": "L3",
        "chain": {
            "L3": _fmt_l3(l3),
            "L2": [_fmt_l2(r) for r in l2_records],
            "L1": [_fmt_l1(r) for r in l1_records],
            "L0": l0_records,
        },
        "stats": {
            "l2_count": len(l2_records),
            "l1_count": len(l1_records),
            "l0_count": len(l0_records),
        }
    }


def _chain_from_l2(l2_id: str) -> dict:
    l2 = _fetch_l2(l2_id)
    if not l2:
        return {"error": f"L2 id={l2_id} 不存在"}

    l1_ids = _parse_json_ids(l2.get("L1_ids"))

    # 同时用 source_l0_ids 直接拉 L0（如果有的话）
    direct_l0_ids = [
        int(i) for i in _parse_json_ids(l2.get("source_l0_ids"))
        if str(i).isdigit()
    ]

    l1_records, l0_via_l1 = _resolve_l1_to_l0(l1_ids)

    # 合并两路 L0，去重
    all_l0_ids = list({r["fact_id"] for r in l0_via_l1} | set(direct_l0_ids))
    l0_records = _fetch_l0_batch(all_l0_ids)

    return {
        "query_id":    l2_id,
        "query_layer": "L2",
        "chain": {
            "L2": [_fmt_l2(l2)],
            "L1": [_fmt_l1(r) for r in l1_records],
            "L0": l0_records,
        },
        "stats": {
            "l1_count": len(l1_records),
            "l0_count": len(l0_records),
        }
    }


def _chain_from_l1(l1_id: str) -> dict:
    l1 = _fetch_l1(l1_id)
    if not l1:
        return {"error": f"L1 id={l1_id} 不存在"}

    _, l0_records = _resolve_l1_to_l0([l1_id])

    return {
        "query_id":    l1_id,
        "query_layer": "L1",
        "chain": {
            "L1": [_fmt_l1(l1)],
            "L0": l0_records,
        },
        "stats": {
            "l0_count": len(l0_records),
        }
    }


# ════════════════════════════════════════════════════════════════
# 格式化工具
# ════════════════════════════════════════════════════════════════

def _fmt_l3(r: dict) -> dict:
    return {
        "id":         r.get("id"),
        "dimension":  r.get("dimension"),
        "content":    r.get("content"),
        "confidence": r.get("confidence"),
        "status":     r.get("status"),
        "confirmed_at": r.get("confirmed_at"),
    }


def _fmt_l2(r: dict) -> dict:
    return {
        "id":          r.get("id"),
        "chunk_name":  r.get("chunk_name"),
        "context":     r.get("context"),
        "category":    r.get("category"),
        "L1_ids":      _parse_json_ids(r.get("L1_ids")),
        "source_l0_ids": _parse_json_ids(r.get("source_l0_ids")),
        "status":      r.get("status"),
    }


def _fmt_l1(r: dict) -> dict:
    return {
        "id":               r.get("id"),
        "type":             r.get("type"),
        "category":         r.get("category"),
        "suggested_summary": r.get("suggested_summary"),
        "reason":           r.get("reason"),
        "confidence":       r.get("confidence"),
        "new_fact_ids":     _parse_json_ids(r.get("new_fact_ids")),
        "related_old_fact_ids": _parse_json_ids(r.get("related_old_fact_ids")),
        "recall_sources":   _parse_json_ids(r.get("recall_sources") or "[]"),
        "status":           r.get("status"),
    }


# ════════════════════════════════════════════════════════════════
# 自动识别层级
# ════════════════════════════════════════════════════════════════

def _detect_layer(id_str: str) -> str:
    """根据 ID 前缀推断层级"""
    s = id_str.lower()
    if s.startswith("l3"):
        return "L3"
    if s.startswith("l2_"):
        return "L2"
    if s.startswith("p") or s.startswith("l1"):
        return "L1"
    # 尝试数字 → L0 fact
    if s.isdigit():
        return "L0"
    # fallback：查库
    conn = _conn()
    if conn.execute("SELECT 1 FROM L3_persona      WHERE id=?", (id_str,)).fetchone():
        conn.close(); return "L3"
    if conn.execute("SELECT 1 FROM L2_chunks        WHERE id=?", (id_str,)).fetchone():
        conn.close(); return "L2"
    if conn.execute("SELECT 1 FROM L1_categorized   WHERE id=?", (id_str,)).fetchone():
        conn.close(); return "L1"
    conn.close()
    return "UNKNOWN"


# ════════════════════════════════════════════════════════════════
# 对外主入口
# ════════════════════════════════════════════════════════════════

def drill_down(id_str: str) -> dict[str, Any]:
    """
    传入任意层 ID，自动识别层级并向下追溯到 L0。
    返回结构化 dict，适合 Agent 程序调用。
    """
    layer = _detect_layer(id_str)

    if layer == "L3":
        return _chain_from_l3(id_str)
    elif layer == "L2":
        return _chain_from_l2(id_str)
    elif layer == "L1":
        return _chain_from_l1(id_str)
    elif layer == "L0":
        fact_id = int(id_str)
        facts   = _fetch_l0_batch([fact_id])
        return {
            "query_id":    id_str,
            "query_layer": "L0",
            "chain":       {"L0": facts},
            "stats":       {"l0_count": len(facts)},
        }
    else:
        return {"error": f"无法识别 ID：{id_str}，请确认是否存在于 L1/L2/L3 表"}


def drill_down_report(id_str: str) -> str:
    """
    同 drill_down()，但返回格式化字符串。
    适合直接注入 Agent prompt，让 LLM 判断记忆是否仍有效。

    示例注入：
        context = drill_down_report("l3001")
        prompt  = f"请判断以下记忆结论是否仍然有效：\\n{context}"
    """
    result = drill_down(id_str)

    if "error" in result:
        return f"[drill_down] ❌ {result['error']}"

    layer = result["query_layer"]
    chain = result["chain"]
    stats = result.get("stats", {})
    lines = [
        f"[追踪链] 查询: {id_str}  起始层: {layer}",
        f"{'─' * 50}",
    ]

    # L3
    if "L3" in chain and chain["L3"]:
        l3 = chain["L3"]
        lines += [
            f"▌ L3 Persona",
            f"  维度   : {l3.get('dimension', '')}",
            f"  结论   : {l3.get('content', '')}",
            f"  置信度 : {l3.get('confidence', '')}",
            "",
        ]

    # L2
    if chain.get("L2"):
        lines.append(f"▌ L2 场景块 ({len(chain['L2'])} 个)")
        for r in chain["L2"]:
            lines += [
                f"  [{r['id']}] {r.get('chunk_name', '')}",
                f"    分类   : {r.get('category', '')}",
                f"    摘要   : {r.get('context', '')}",
            ]
        lines.append("")

    # L1
    if chain.get("L1"):
        lines.append(f"▌ L1 归类记录 ({len(chain['L1'])} 条)")
        for r in chain["L1"]:
            lines += [
                f"  [{r['id']}] {r.get('category', '')} | 置信度={r.get('confidence', '')}",
                f"    摘要   : {r.get('suggested_summary', '')}",
                f"    原因   : {r.get('reason', '')}",
                f"    来源   : new={r.get('new_fact_ids',[])} related={r.get('related_old_fact_ids',[])}",
            ]
        lines.append("")

    # L0
    if chain.get("L0"):
        lines.append(f"▌ L0 原始事实 ({len(chain['L0'])} 条)")
        for f in chain["L0"]:
            lines.append(f"  [id={f.get('fact_id', f.get('id'))}] {f.get('content', '')}")
        lines.append("")

    lines.append(f"{'─' * 50}")
    lines.append(
        f"汇总: L2={stats.get('l2_count','?')} L1={stats.get('l1_count','?')} "
        f"L0={stats.get('l0_count','?')}"
    )

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════

def cli():
    if len(sys.argv) < 2:
        print("用法: python3 drill_down.py <id>")
        print("示例: python3 drill_down.py l3001")
        print("      python3 drill_down.py L2_p001")
        print("      python3 drill_down.py p001")
        sys.exit(1)

    id_str = sys.argv[1]
    fmt    = sys.argv[2] if len(sys.argv) > 2 else "report"

    if fmt == "json":
        result = drill_down(id_str)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(drill_down_report(id_str))


if __name__ == "__main__":
    cli()
