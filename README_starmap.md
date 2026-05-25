# StarMap Memory 使用文档

## 是什么

StarMap（星图记忆）是一个 4 层被动分级记忆系统，建在 Hermes 内置 Holographic Memory（fact_store）之上。

目标：**把碎片化的原始事实（fact）提炼成有结构的知识层次，供 Hermes 在回答时快速召回。**

---

## 四层结构

```
L0  原始事实
    fact_store 里的每条 fact，原样不动
    例：用户偏好简洁回答、GBM已放弃、开盘红API Token过期

L1  分类标签
    给 L0 打标签（category + reason）
    把分散的 fact 归到同一个主题下
    一条 L1 绑定多个 source_l0_ids，可追溯原始内容

L2  场景块
    把相关 L1 合并成一段连贯的总结
    例："开盘红GBM项目" — 包含数据库路径、GBM差值阈值、项目定位等
    供 Hermes 直接召回，有上下文，不碎片

L3  Persona
    每月提炼：Hermes 对用户这个人的稳定判断
    例：用户是A股短线交易者、偏好简洁回答、用UU远程桌面
    不参与日常召回，只在需要理解用户时用
```

---

## 架构图

```
新 fact 写入 fact_store（L0）
         ↓
    seed_expand.py
    ① 算 Embedding（混元 hunyuan-embedding）
    ② 混合召回 top5 相似 fact
    ③ DeepSeek Chat 生成 L1/L2 建议
    ④ 直接 confirmed 写入 DB（无需人工确认）
    ⑤ rebuild L2 FTS 索引
         ↓
┌────────┴────────┐
↓                 ↓
L1_categorized   L2_chunks
（带分类的事实）  （合并后的场景块）
      ↓
  query_starmap_l2_chunks（FTS5 全文索引）
         ↓
    prefetch 召回
```

---

## 核心组件

| 文件 | 作用 |
|------|------|
| `seed_expand.py` | 新 fact 触发，L1/L2 实时生成并写入 DB |
| `confirm_write.py` | pending 确认 + DB 写入（L1/L2/L3） |
| `starmap_cleanup.py` | 每周六只做 FTS 重建 |
| `store.py` | MemoryStore，新增 `rebuild_l2_chunks_fts()` + `query_starmap_l2_chunks()` |
| `__init__.py` | HolographicMemoryProvider.prefetch() 改为 L2 优先召回 |
| `embed_utils.py` | 混元 embedding（向量 + FTS 索引） |
| `llm_client.py` | DeepSeek Chat 调用 |

---

## 使用现状

| 指标 | 数量 |
|------|------|
| L0 facts | 635 条 |
| L1 confirmed | 123 条 |
| L2 confirmed chunks | 332 条 |
| L3 persona | 少量 |

---

## 当前召回逻辑

```
prefetch(query)
    → L2 查询（FTS5，limit=5）
    → 有结果 → 返回 StarMap L2（只召 L2）
    → 无结果 → 回落 L0 raw facts（limit=5）
```

L2 优先，确保 Hermes 拿到的都是有结构的提炼内容，不是散碎片。

---

## 优点

1. **召回质量高** — L2 是已确认的提炼结论，不是临时拼凑，模型不需要现场推理关联
2. **防止幻觉** — 碎片每次召回数和顺序不确定，L2 是固化内容
3. **长会话省 token** — 不必每次把全部 L0 塞进 prompt
4. **上下文清晰** — L2 块有 chunk_name（标题）和 content（总结），一目了然
5. **可追溯** — L2 块内含 source_l0_ids，可回查原始 fact

---

## 瓶颈

| 瓶颈 | 说明 | 当前状态 |
|------|------|----------|
| FTS 全量重建 | `rebuild_l2_chunks_fts()` 每次 DROP + 全表重建，L2 到 1000+ 条会变慢 | 目前 332 条，无感 |
| Embedding API 依赖 | 每次 seed_expand 调混元 embedding | 100万token/1元，消耗极少 |
| LLM 质量依赖 | DeepSeek 生成 L1/L2 建议的质量不确定 | 已改为直接 confirmed，无人工审核 |
| L2 块数量上限 | limit=5 防止超 token，但可能遗漏相关内容 | L2 为空时回落 L0，不完全漏掉 |

---

## 下一步计划

### 短期（近期可做）
1. **增量 FTS 索引** — 改为只对新增 L2 做 INSERT，不全量重建
2. **调整 L2召回 limit** — 根据实际情况从 5 调到 3 或 8
3. **L2 召回结果直接追溯 L0** — 召到 L2 块后，展示其 source_l0_ids 对应的原始 fact

### 中期（有意义但非紧急）
4. **L3 召回接入 prefetch** — 日常召回也考虑 L3 persona 层
5. **错题本机制** — 发现 L2 召回错误时，自动标记并在下一次 seed_expand 时重新生成
6. **merge 阈值优化** — 调整 L1→L2 合并的条件，减少低质量 chunk

### 长期（如果 StarMap 规模持续增长）
7. **向量索引替代 FTS** — L2 数量大后，纯 FTS 精度不如向量相似度
8. **多跳推理** — 当单一 L2 块不够时，通过 L1_ids 做关联召回

---

## 与 Hermes 内置 memory 的关系

- **fact_store（Holographic）** — 原始 L0，Holographic 自带压缩和检索
- **StarMap** — 在 L0 之上建 L1/L2/L3 层，做二次提炼
- **prefetch** — 两者同时存在，L2 优先，L0 兜底