"""
monthly_persona.py
月度回顾：每月1号 cron 触发
  1. 检查 L0 >= 100 且有 confirmed L1 记录
  2. 扫描所有 confirmed L1_categorized
  3. 交给 LLM → 提炼 L3 Persona 建议
  4. 追加到 pending_review.json
"""

import sqlite3
import json
import os
from datetime import datetime

from llm_client import call_llm
from confirm_write import append_to_pending

DB_PATH      = os.environ.get("MEMORY_DB", "/home/ye/.hermes/memory_store.db")
L0_THRESHOLD = 100


# ── 数据读取 ────────────────────────────────────────────────────

def get_l0_count() -> int:
    conn = sqlite3.connect(DB_PATH)
    count = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    conn.close()
    return count


def get_confirmed_l1() -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    rows = cur.execute("""
        SELECT id, category, suggested_summary, reason, confidence
        FROM L1_categorized
        WHERE status = 'confirmed'
        ORDER BY confirmed_at DESC
    """).fetchall()
    conn.close()
    return [
        {
            "id": r[0], "category": r[1],
            "summary": r[2], "reason": r[3], "confidence": r[4],
        }
        for r in rows
    ]


SYSTEM_PROMPT = """你是一个用户画像分析助手。
用户会给你一批已确认的记忆分类记录（L1）。
请从中提炼出用户的核心人格特征、偏好和习惯，形成 Persona（L3）建议。

维度参考（可扩展）：
- 工具偏好：喜欢哪类工具，抗拒什么
- 交互风格：喜欢什么样的沟通方式
- 交易习惯：如果有交易相关记录
- 工作方式：处理问题的思维模式
- 核心价值观：做决定的底层逻辑

请只返回 JSON 数组，每个元素是一个 Persona 条目。格式：
[
  {
    "dimension": "工具偏好",
    "content": "偏好轻量方案，抗拒引入重型依赖，优先考虑零维护成本的选择",
    "source_L1_ids": ["p001", "p003"],
    "confidence": 0.88
  }
]
不要返回任何 JSON 以外的内容。"""


def gen_proposal_id(prefix: str = "l3") -> str:
    ts = datetime.now().strftime("%f")   # microsecond，6位
    return f"{prefix}{ts}"


# ── 主入口 ──────────────────────────────────────────────────────

def run():
    # ── 防重复触发检查 ─────────────────────────────────────────
    conn_check = sqlite3.connect(DB_PATH)
    last_trigger = conn_check.execute("""
        SELECT l1_count, trigger_status, triggered_at FROM l3_trigger_log
        ORDER BY id DESC LIMIT 1
    """).fetchone()
    conn_check.close()

    if last_trigger and last_trigger[1] == "completed":
        # 只在当月已运行过才跳过；新月份重新触发
        triggered_month = last_trigger[2][:7]  # "YYYY-MM"
        current_month   = datetime.now().strftime("%Y-%m")
        if triggered_month == current_month:
            print(f"[monthly] ⚠️ 本月（{current_month}）已运行过，跳过")
            return
        print(f"[monthly] 上次运行 {triggered_month}，当前 {current_month}，继续")
    elif last_trigger and last_trigger[1] == "pending":
        print(f"[monthly] ⚠️ 已存在 pending 记录，跳过")
        return

    l0_count = get_l0_count()
    print(f"[monthly] L0 count={l0_count}")

    if l0_count < L0_THRESHOLD:
        print(f"[monthly] L0 不足 {L0_THRESHOLD} 条，跳过月度回顾")
        return

    confirmed_l1 = get_confirmed_l1()
    if not confirmed_l1:
        print("[monthly] 没有 confirmed L1 记录，跳过")
        return

    print(f"[monthly] 读取到 {len(confirmed_l1)} 条 confirmed L1")

    # ── 记录触发 ───────────────────────────────────────────────
    conn_log = sqlite3.connect(DB_PATH)
    l1_ids_json = json.dumps([l1["id"] for l1 in confirmed_l1])
    l1_count = len(confirmed_l1)
    trigger_status = "pending"
    conn_log.execute("""
        INSERT INTO l3_trigger_log (l1_id_set, l1_count, trigger_status)
        VALUES (?, ?, ?)
    """, (l1_ids_json, l1_count, trigger_status))
    conn_log.commit()
    trigger_id = conn_log.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn_log.close()

    # 构造 LLM 输入
    l1_text = json.dumps(confirmed_l1, ensure_ascii=False, indent=2)
    raw = call_llm(SYSTEM_PROMPT, f"以下是用户的 L1 记忆记录：\n{l1_text}",
                   max_tokens=2000)

    def _update_status(status: str):
        conn_log2 = sqlite3.connect(DB_PATH)
        conn_log2.execute("UPDATE l3_trigger_log SET trigger_status=? WHERE id=?",
                         (status, trigger_id))
        conn_log2.commit()
        conn_log2.close()

    if not raw:
        print("[monthly] ❌ LLM 返回空，跳过")
        _update_status("failed")
        return

    clean = raw.replace("```json", "").replace("```", "").strip()
    try:
        proposals_raw = json.loads(clean)
    except json.JSONDecodeError as e:
        print(f"[monthly] ❌ JSON 解析失败: {e}\n原文: {raw}")
        _update_status("failed")
        return

    if not isinstance(proposals_raw, list):
        print("[monthly] ❌ LLM 返回格式错误，期望列表")
        _update_status("failed")
        return

    # 包装成标准 proposal 格式
    proposals = []
    for p in proposals_raw:
        pid = gen_proposal_id()
        proposals.append({
            "id": pid,
            "dimension": p.get("dimension", ""),
            "content": p.get("content", ""),
            "source_L1_ids": p.get("source_L1_ids", []),
            "confidence": p.get("confidence", 0.0),
            "status": "pending",
            "source": "monthly",
            "generated_at": datetime.now().isoformat(),
        })

    # 追加到 pending_review.json
    for p in proposals:
        append_to_pending(p, trigger_type="monthly", l0_count=l0_count,
                          layer="L3")

    print(f"[monthly] ✅ 生成 {len(proposals)} 条 L3 Persona 建议")
    _update_status("completed")
    return 0


if __name__ == "__main__":
    run()
