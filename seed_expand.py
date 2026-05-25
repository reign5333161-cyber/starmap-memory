"""
seed_expand.py
种子膨胀：新 fact 写入 L0 后触发
  1. L0 >= 100 才进入进阶模式
  2. 算 Embedding（带缓存）+ 写入 FTS 索引
  3. 混合召回（向量 + BM25 RRF），相似度 > 0.8，top 5
  4. 新 + top5 → LLM（混元/DeepSeek）→ 生成 L1/L2 建议
  5. 追加到 pending_review.json
"""

import sqlite3
import json
import os
import time
from datetime import datetime
from pathlib import Path

from llm_client import call_llm
from embed_utils import (
    get_or_create_vector,
    search_hybrid,
    fetch_fact_contents,
    index_fact_to_fts,
)
from confirm_write import append_to_pending, confirm_all

DB_PATH      = os.environ.get("MEMORY_DB", "/home/ye/.hermes/memory_store.db")
THRESHOLD    = 0.8
TOP_K        = 5
L0_THRESHOLD = 100


def get_l0_count() -> int:
    conn  = sqlite3.connect(DB_PATH)
    count = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    conn.close()
    return count


def gen_proposal_id(prefix: str = "p") -> str:
    ts = int(time.time() * 1000) % 1000000
    return f"{prefix}{ts:06d}"


# ── LLM 归类 Prompt ──────────────────────────────────────────────

SYSTEM_PROMPT = """你是一个记忆归类助手。
用户会给你一条"新事实"和若干条"相关旧事实"（含来源：向量相似 / BM25关键词 / 双命中）。
判断这些事实能否合并成一个有意义的"场景块（L2）"。

只返回 JSON，不要任何额外文字：
{
  "can_merge": true 或 false,
  "type": "merge_to_l2" 或 "label_only",
  "category": "Project/XXX 或 user_pref 或 tool 或 general",
  "suggested_summary": "一句话描述（Markdown 格式，可加 **重点**）",
  "chunk_name": "场景块名称",
  "reason": "为什么这些事实相关",
  "confidence": 0到1的浮点数
}
can_merge=false 时，chunk_name 和 suggested_summary 填空字符串。"""


def llm_expand(
    new_fact_id: int,
    new_content: str,
    similar: list[dict],
    old_contents: dict[int, str],
) -> dict | None:
    old_lines = "\n".join(
        f"- [id={s['fact_id']}, 来源={s.get('source','?')}, "
        f"rrf={s.get('rrf_score',0):.4f}] {old_contents.get(s['fact_id'], '')}"
        for s in similar
    )
    user_msg = f"""新事实（id={new_fact_id}）：
{new_content}

相关旧事实（混合召回）：
{old_lines}
"""
    raw = call_llm(SYSTEM_PROMPT, user_msg, max_tokens=1000)
    if not raw:
        return None

    clean = raw.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(clean)
    except json.JSONDecodeError as e:
        print(f"[llm] ❌ JSON 解析失败: {e}\n原文: {raw}")
        return None

    # 收集所有 L0 来源 ID（新 fact + 相关旧 fact）
    source_l0_ids = [new_fact_id] + [s["fact_id"] for s in similar]

    pid = gen_proposal_id()
    return {
        "id":                   pid,
        "type":                 data.get("type", "label_only"),
        "new_fact_ids":         [new_fact_id],
        "related_old_fact_ids": [s["fact_id"] for s in similar],
        "similarity_scores":    [s.get("rrf_score", 0) for s in similar],
        "recall_sources":       [s.get("source", "?") for s in similar],  # 来源追踪
        "source_l0_ids":        source_l0_ids,                             # L2→L0 追踪链
        "reason":               data.get("reason", ""),
        "suggested_summary":    data.get("suggested_summary", ""),
        "chunk_name":           data.get("chunk_name", ""),
        "category":             data.get("category", "general"),
        "confidence":          data.get("confidence", 0.0),
        "can_merge":           data.get("can_merge", False),
        "status":              "confirmed",
        "source":              "seed_expand",
        "generated_at":         datetime.now().isoformat(),
    }


# ── 主入口 ──────────────────────────────────────────────────────

def run(fact_id: int, content: str):
    """
    新 fact 写入 L0 后调用。
    fact_id : 已写入 facts 表的 id
    content : fact 原文
    """
    # 无论是否进阶模式，都写入 FTS 索引（供未来 BM25 使用）
    index_fact_to_fts(fact_id, content)

    l0_count = get_l0_count()
    print(f"[seed_expand] L0 count={l0_count}, threshold={L0_THRESHOLD}")

    if l0_count < L0_THRESHOLD:
        print(f"[seed_expand] 冷启动阶段，跳过分析（还差 {L0_THRESHOLD - l0_count} 条）")
        return

    # 1. 向量化（带缓存）
    vec = get_or_create_vector(fact_id, content)
    if vec is None:
        print("[seed_expand] ❌ Embedding 失败，跳过")
        return

    # 2. 混合召回（向量 + BM25 RRF）
    similar = search_hybrid(
        query_text=content,
        query_vec=vec,
        exclude_ids=[fact_id],
        threshold=THRESHOLD,
        top_k=TOP_K,
    )
    print(f"[seed_expand] 混合召回 {len(similar)} 条相关 facts")

    # 召回不足时降级阈值重新搜索（ reviewer: 当结果≤2条时触发）
    if 0 < len(similar) <= 2:
        print(f"[seed_expand] ⚠️  召回仅 {len(similar)} 条，降至 threshold=0.6 重新搜索")
        fallback = search_hybrid(
            query_text=content,
            query_vec=vec,
            exclude_ids=[fact_id] + [s["fact_id"] for s in similar],
            threshold=0.6,
            top_k=TOP_K,
        )
        # 追加新结果（去重已由 exclude_ids 处理）
        existing_ids = {s["fact_id"] for s in similar}
        for item in fallback:
            if item["fact_id"] not in existing_ids:
                similar.append(item)
        print(f"[seed_expand] ✅ 降级后共 {len(similar)} 条")

    if not similar:
        print("[seed_expand] 没有相关记忆，不生成建议")
        return

    # 3. 拉取旧 fact 原文
    old_contents = fetch_fact_contents([s["fact_id"] for s in similar])

    # 4. LLM 归类
    proposal = llm_expand(fact_id, content, similar, old_contents)
    if proposal is None:
        print("[seed_expand] ❌ LLM 返回空，跳过")
        return

    # 5. 写入 pending_review.json → 直接 confirmed 入 DB
    append_to_pending(proposal, trigger_type="seed_expand", l0_count=l0_count)
    print(f"[seed_expand] ✅ 建议已写入 pending_review.json (id={proposal['id']})")
    written = confirm_all()
    print(f"[seed_expand] ✅ 已确认写入 DB (written={written})")

    # 6. 标记 L2 FTS 索引为 dirty（下次 prefetch 时按需重建，避免每次强制全量重建）
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        sys.path.insert(0, "/home/ye/.hermes/hermes-agent")
        from plugins.memory.holographic.store import MemoryStore
        MemoryStore().mark_l2_fts_dirty()
        print("[seed_expand] ✅ L2 FTS 已标记为 dirty，将在下次 prefetch 时重建")
    except Exception as e:
        print(f"[seed_expand] ⚠️  mark_l2_fts_dirty 失败（不影响 DB 数据）: {e}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3:
        run(int(sys.argv[1]), sys.argv[2])
    else:
        print("用法: python seed_expand.py <fact_id> <content>")
