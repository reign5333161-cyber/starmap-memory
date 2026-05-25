[README.md](https://github.com/user-attachments/files/28207355/README.md)
# StarMap Memory

A 4-tier passive hierarchical memory system built on top of Hermes' built-in Holographic Memory (`fact_store`).

**Goal:** Transform fragmented raw facts into structured knowledge layers, enabling fast and context-aware retrieval for LLM agents.

## Architecture

```
L0 ─ Raw Facts
     Original entries in fact_store, untouched.

L1 ─ Categorized Tags
     Clustered L0 facts under topic labels with traceable source IDs.

L2 ─ Scene Chunks
     Coherent summaries merging related L1 entries.
     Directly served to the agent as retrieval context.

L3 ─ Persona
     Monthly distilled stable judgments about the user.
     Used for long-term personalization, not daily recall.
```

## How It Works

```
New fact → fact_store (L0)
    ↓
seed_expand.py
    → Compute embedding (Tencent Hunyuan)
    → Hybrid recall top-5 similar facts
    → DeepSeek Chat generates L1/L2 suggestions
    → Auto-confirm and write to DB
    → Rebuild L2 FTS index
    ↓
prefetch(query)
    → L2 FTS5 query (limit=5)
    → Hit → return StarMap L2 chunks
    → Miss → fallback to L0 raw facts
```

## Project Structure

| File | Role |
|------|------|
| `seed_expand.py` | Triggered on new fact; generates L1/L2 in real-time |
| `confirm_write.py` | Pending confirmation + DB writes (L1/L2/L3) |
| `starmap_cleanup.py` | Weekly FTS index rebuild |
| `store.py` | MemoryStore with `rebuild_l2_chunks_fts()` and `query_starmap_l2_chunks()` |
| `__init__.py` | `HolographicMemoryProvider.prefetch()` — L2-priority recall |
| `embed_utils.py` | Hunyuan embedding (vector + FTS indexing) |
| `llm_client.py` | DeepSeek Chat API client |
| `drill_down.py` | Trace L2 chunks back to source L0 facts |
| `00_init_db.py` | Database schema initialization |
| `01_batch_init.py` | Batch seeding and re-processing scripts |
| `monthly_persona.py` | Monthly L3 persona distillation |

## Key Advantages

- **High-quality retrieval** — L2 chunks are pre-condensed conclusions, not ad-hoc assemblies
- **Hallucination resistance** — Solidified content prevents retrieval randomness
- **Token efficient** — Avoids dumping all L0 facts into context
- **Traceable** — Every L2 chunk links back to source L0 fact IDs
- **Context-aware** — Chunks carry titles and summaries for clear interpretation

## Current Scale

| Tier | Count |
|------|-------|
| L0 facts | 635 |
| L1 confirmed | 123 |
| L2 chunks | 332 |
| L3 persona | Minimal |

## Status & Roadmap

- [x] L2-priority retrieval
- [x] Auto-confirm pipeline (no manual review)
- [ ] Incremental FTS index (vs. full rebuild)
- [ ] L3 persona integration into prefetch
- [ ] Vector index as FTS replacement at scale
- [ ] Multi-hop reasoning across chunks

## License

MIT
