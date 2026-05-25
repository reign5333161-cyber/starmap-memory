"""
embed_utils.py
向量化工具：Embedding + SQLite 缓存 + 混合召回（向量 + BM25 RRF）

Embedding API 配置：
  API Key  : HUNYUAN_API_KEY（腾讯混元）
  端点     : https://api.hunyuan.cloud.tencent.com/v1/embeddings
  模型     : hunyuan-embedding（1024维）
"""

import sqlite3
import json
import hashlib
import os
import time
import requests
import numpy as np
from typing import Optional

DB_PATH = os.environ.get("MEMORY_DB", "/home/ye/.hermes/memory_store.db")

# Embedding 专用配置（混元 Hunyuan Embedding）
EMBED_API_KEY = (
    os.environ.get("HUNYUAN_API_KEY")
    or os.environ.get("DEEPSEEK_API_KEY")
    or ""
)
EMBED_URL   = os.environ.get("EMBED_API_URL", "https://api.hunyuan.cloud.tencent.com/v1/embeddings")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "hunyuan-embedding")


# ── 基础工具 ─────────────────────────────────────────────────────

def _hash(text: str) -> str:
    return hashlib.md5(text.strip().encode()).hexdigest()

def _vec_to_blob(vec: list[float]) -> bytes:
    return json.dumps(vec).encode()

def _blob_to_vec(blob: bytes) -> np.ndarray:
    return np.array(json.loads(blob.decode()), dtype=np.float32)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ── Embedding API ────────────────────────────────────────────────

def call_embedding_api(text: str, retry: int = 3) -> Optional[list[float]]:
    if not EMBED_API_KEY:
        raise ValueError("请设置 DEEPSEEK_API_KEY 或 HUNYUAN_API_KEY")

    headers = {
        "Authorization": f"Bearer {EMBED_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"model": EMBED_MODEL, "input": text}

    for attempt in range(retry):
        try:
            resp = requests.post(EMBED_URL, headers=headers,
                                 json=payload, timeout=30)
            resp.raise_for_status()
            return resp.json()["data"][0]["embedding"]
        except Exception as e:
            print(f"[embed] ⚠️  第{attempt+1}次失败: {e}")
            if attempt < retry - 1:
                time.sleep(2 ** attempt)
    return None


# ── 向量缓存 ─────────────────────────────────────────────────────

def get_or_create_vector(fact_id: int, content: str) -> Optional[np.ndarray]:
    """hash 命中直接返回缓存，否则调 API 并写入"""
    h    = _hash(content)
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()

    row = cur.execute(
        "SELECT vector, content_hash FROM facts_vectors WHERE fact_id = ?",
        (fact_id,)
    ).fetchone()

    if row and row[1] == h:
        conn.close()
        return _blob_to_vec(row[0])

    vec = call_embedding_api(content)
    if vec is None:
        conn.close()
        return None

    blob = _vec_to_blob(vec)
    cur.execute("""
        INSERT INTO facts_vectors (fact_id, content_hash, vector, dim, updated_at)
        VALUES (?, ?, ?, ?, datetime('now'))
        ON CONFLICT(fact_id) DO UPDATE SET
            content_hash = excluded.content_hash,
            vector       = excluded.vector,
            dim          = excluded.dim,
            updated_at   = excluded.updated_at
    """, (fact_id, h, blob, len(vec)))
    conn.commit()
    conn.close()

    print(f"[embed] ✅ fact_id={fact_id} 向量已{'更新' if row else '写入'}缓存")
    return np.array(vec, dtype=np.float32)


# ── 向量搜索 ─────────────────────────────────────────────────────

def search_vector(
    query_vec: np.ndarray,
    exclude_ids: list[int],
    threshold: float = 0.8,
    top_k: int = 20,           # 给 RRF 多拿一些候选
) -> list[dict]:
    """余弦相似度 >= threshold，返回 top_k，含 rank 字段"""
    conn = sqlite3.connect(DB_PATH)
    if exclude_ids:
        ph = ",".join("?" * len(exclude_ids))
        rows = conn.execute(
            f"SELECT fact_id, vector FROM facts_vectors WHERE fact_id NOT IN ({ph})",
            exclude_ids
        ).fetchall()
    else:
        # 空 exclude_ids 时查全表，不要 NOT IN (NULL)
        rows = conn.execute("SELECT fact_id, vector FROM facts_vectors").fetchall()
    conn.close()

    results = []
    for fact_id, blob in rows:
        sim = cosine_similarity(query_vec, _blob_to_vec(blob))
        if sim >= threshold:
            results.append({"fact_id": fact_id, "similarity": round(sim, 4)})

    results.sort(key=lambda x: x["similarity"], reverse=True)
    # 注入向量排名
    for rank, item in enumerate(results[:top_k], start=1):
        item["vector_rank"] = rank
    return results[:top_k]


# ── BM25 全文检索（SQLite FTS5）────────────────────────────────

def _build_fts_query(text: str) -> str:
    """
    为 trigram 分词器构建 FTS 查询。
    - 提取长度 >= 3 的连续中文片段和英文单词
    - 多个词用 OR 连接，扩大召回
    - 若无符合条件的词，返回空字符串（跳过 BM25）
    - FTS5 特殊字符（"、-、(、) 等）转义，防止 MATCH 报错
    """
    import re
    tokens = []

    # 英文单词（>=3字符）
    eng = re.findall(r'[A-Za-z][A-Za-z0-9_]{2,}', text)
    tokens.extend(eng)

    # 中文连续片段，按非中文字符切割，取长度>=3的
    zh_chunks = re.split(r'[^\u4e00-\u9fff]+', text)
    for chunk in zh_chunks:
        if len(chunk) >= 3:
            tokens.append(chunk)
        elif len(chunk) == 2:
            # 2字中文加首字扩展成3字：取前两字+下一中文（已在chunk里找不到），
            # 直接用2字+通配，trigram不支持通配，跳过
            pass

    # 去重，保持顺序。trigram 分词器对单引号不敏感，
    # 不需要额外转义，直接拼接即可。
    seen, unique = set(), []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    return " OR ".join(unique)


def search_bm25(
    query: str,
    exclude_ids: list[int],
    top_k: int = 20,
) -> list[dict]:
    """
    用 FTS5 BM25 搜索 facts_fts 表。
    返回 [{"fact_id": int, "bm25_score": float, "bm25_rank": int}]
    bm25() 返回负数，越小越相关，取反后越大越好。
    """
    # 构建 trigram 兼容的查询词
    safe_query = _build_fts_query(query)
    if not safe_query:
        print("[bm25] ⚠️  无有效查询词（需要>=3字符），跳过 BM25")
        return []

    conn = sqlite3.connect(DB_PATH)

    # 检查 FTS 表是否存在
    has_fts = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='facts_fts'"
    ).fetchone()
    if not has_fts:
        conn.close()
        print("[bm25] ⚠️  facts_fts 表不存在，跳过 BM25")
        return []

    try:
        rows = conn.execute("""
            SELECT f.content_id, -bm25(facts_fts) AS score
            FROM facts_fts f
            WHERE facts_fts MATCH ?
            ORDER BY score DESC
            LIMIT ?
        """, (safe_query, top_k * 3)).fetchall()
    except Exception as e:
        print(f"[bm25] ⚠️  FTS 查询失败: {e}")
        conn.close()
        return []

    conn.close()

    results = []
    seen = set(exclude_ids)
    rank = 1
    for content_id, score in rows:
        try:
            fid = int(content_id)
        except (TypeError, ValueError):
            continue
        if fid in seen:
            continue
        seen.add(fid)
        results.append({
            "fact_id":    fid,
            "bm25_score": round(score, 4),
            "bm25_rank":  rank,
        })
        rank += 1
        if rank > top_k:
            break

    return results


# ── RRF 融合 ─────────────────────────────────────────────────────

def rrf_fusion(
    vector_results: list[dict],
    bm25_results:   list[dict],
    top_k: int = 5,
    k: int = 60,               # RRF 平滑常数，标准值 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion：
      score(d) = Σ 1/(k + rank_i(d))
    两路结果合并，按融合分排序，取 top_k。
    """
    scores: dict[int, float] = {}

    for item in vector_results:
        fid  = item["fact_id"]
        rank = item.get("vector_rank", 1)
        scores[fid] = scores.get(fid, 0.0) + 1.0 / (k + rank)

    for item in bm25_results:
        fid  = item["fact_id"]
        rank = item.get("bm25_rank", 1)
        scores[fid] = scores.get(fid, 0.0) + 1.0 / (k + rank)

    # 构建元数据索引，方便回填
    vec_meta  = {x["fact_id"]: x for x in vector_results}
    bm25_meta = {x["fact_id"]: x for x in bm25_results}

    merged = []
    for fid, rrf_score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]:
        entry = {
            "fact_id":   fid,
            "rrf_score": round(rrf_score, 6),
            # 保留原始分供调试
            "similarity":  vec_meta[fid]["similarity"]  if fid in vec_meta  else None,
            "bm25_score":  bm25_meta[fid]["bm25_score"] if fid in bm25_meta else None,
            "source":      (
                "both"   if (fid in vec_meta and fid in bm25_meta) else
                "vector" if fid in vec_meta else "bm25"
            ),
        }
        merged.append(entry)

    return merged


# ── 混合搜索（对外主入口）────────────────────────────────────────

def search_hybrid(
    query_text: str,
    query_vec:  np.ndarray,
    exclude_ids: list[int],
    threshold: float = 0.8,
    top_k: int = 5,
) -> list[dict]:
    """
    向量 + BM25 混合召回，RRF 融合后返回 top_k 条。
    每条包含: fact_id, rrf_score, similarity, bm25_score, source
    """
    vec_results  = search_vector(query_vec, exclude_ids, threshold, top_k=top_k*4)
    bm25_results = search_bm25(query_text, exclude_ids, top_k=top_k*4)

    # 两路均为空 → 降级到纯向量结果（忽略 threshold 限制放宽到 0.6）
    if not vec_results and not bm25_results:
        print("[hybrid] ⚠️  两路均无结果，降级纯向量 threshold=0.6")
        vec_results = search_vector(query_vec, exclude_ids, threshold=0.6, top_k=top_k)
        return [
            {
                "fact_id":    x["fact_id"],
                "rrf_score":  x["similarity"],
                "similarity": x["similarity"],
                "bm25_score": None,
                "source":     "vector_fallback",
            }
            for x in vec_results[:top_k]
        ]

    fused = rrf_fusion(vec_results, bm25_results, top_k=top_k)

    v_cnt = sum(1 for x in fused if x["source"] == "vector")
    b_cnt = sum(1 for x in fused if x["source"] == "bm25")
    both  = sum(1 for x in fused if x["source"] == "both")
    print(f"[hybrid] 融合结果: {len(fused)} 条 "
          f"(向量独占={v_cnt}, BM25独占={b_cnt}, 双命中={both})")

    return fused


# ── 工具函数 ─────────────────────────────────────────────────────

def fetch_fact_contents(fact_ids: list[int]) -> dict[int, str]:
    if not fact_ids:
        return {}
    conn = sqlite3.connect(DB_PATH)
    ph   = ",".join("?" * len(fact_ids))
    rows = conn.execute(
        f"SELECT fact_id, content FROM facts WHERE fact_id IN ({ph})", fact_ids
    ).fetchall()
    conn.close()
    return {r[0]: r[1] for r in rows}


def index_fact_to_fts(fact_id: int, content: str):
    """把一条 fact 写入 FTS 索引（新增时调用）"""
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            "INSERT OR IGNORE INTO facts_fts(content, content_id) VALUES (?, ?)",
            (content, str(fact_id))
        )
        conn.commit()
    except Exception as e:
        print(f"[fts] ⚠️  写入 FTS 失败 fact_id={fact_id}: {e}")
    finally:
        conn.close()
