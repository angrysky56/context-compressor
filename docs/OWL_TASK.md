# Task: Embedding Search Fixes (context-compressor)

Work through these in order. Do not skip acceptance criteria.
Constraints: ruff clean, no `print()` to stdout anywhere (stdio MCP server —
stdout is the JSON-RPC channel; use `logging` to stderr), uv toolchain,
Python 3.12+ typing (`list`, `dict`, `X | None`).

## 1. Use Qwen3-Embedding correctly
The model card specifies **last-token (EOS) pooling with left padding**,
not mean pooling, and **instruction-prefixed queries** for retrieval:
- Set `tokenizer.padding_side = "left"` before batch tokenization.
- Pool: take the hidden state of the last non-pad token per sequence
  (with left padding this is simply `last_hidden_state[:, -1]`).
- Queries get a prefix; documents are embedded bare:
  `f"Instruct: Given a web search query, retrieve relevant passages\nQuery: {q}"`
- Add a `is_query: bool = False` param to `embed_texts` to switch modes.
Check the Qwen/Qwen3-Embedding-0.6B model card on Hugging Face and match
its reference usage exactly. Do not guess.

## 2. Honest failure instead of silent fallback
- Remove `except Exception as e: pass` in `_rebuild_search_index`.
- Log the failure via `logging` (stderr) with the exception message.
- Track index state: `_search_index.status` in {"ready", "unavailable"}.
- `search_chunks` must include `"index_status"` in its response so a
  calling agent can distinguish "no matches" from "search is broken".
- Restore TF-IDF as a working fallback path when the embedding model
  fails to load: degrade to TF-IDF search and report
  `index_status: "fallback_tfidf"`. Do not delete working fallbacks.

## 3. Resource handling
- Load the model with `torch_dtype=torch.bfloat16` when CUDA is
  available; float32 on CPU.
- Raise `max_length` from 512 to 4096 for document embedding
  (queries can stay at 512).
- Batch the corpus in `_rebuild_search_index`: embed at most 16 texts
  per forward pass. Never pad the entire corpus in one batch.

## 4. Persist embeddings, stop re-embedding everything
- Store in each chunk's JSON: `embedding` (list[float]),
  `embedding_model` (str), `embedding_dim` (int).
- On rebuild: load stored vectors; only embed chunks that are missing
  a vector OR whose `embedding_model` differs from the active model.
- If stored model != active model for some chunks, re-embed those
  chunks and overwrite. Never mix vectors from different models in
  one similarity matrix.

## 5. Pluggable backend
- `EMBED_BACKEND` env var: `"local"` (default) or `"openrouter"`.
- OpenRouter backend: POST to `https://openrouter.ai/api/v1/embeddings`
  with `Authorization: Bearer $OPENROUTER_API_KEY`, model
  `nvidia/llama-nemotron-embed-vl-1b-v2:free`. Use `httpx` if already
  a dependency; otherwise `urllib.request` from stdlib. No new deps.
- The backend must implement the same `embed_texts(texts, is_query)`
  interface. Network errors follow rule 2: logged, status surfaced,
  TF-IDF fallback engaged.

## 6. One real integration test
Add `tests/test_embedding_search.py`:
- Marked `@pytest.mark.integration`, skipped unless the model is
  available locally (`pytest.importorskip("torch")` + env flag
  `RUN_EMBED_TESTS=1`).
- Compress two short texts with clearly different topics (e.g. one
  about Neo4j graph queries, one about sourdough baking).
- Query "graph database cypher" and assert the Neo4j chunk ranks
  first with a higher score than the baking chunk.
- This test existing and passing is the proof the pooling fix works.

## Acceptance criteria (all must hold)
- [ ] `uv run pytest tests/ -v` passes (67 existing + new test)
- [ ] `RUN_EMBED_TESTS=1 uv run pytest tests/test_embedding_search.py -v` passes
- [ ] `uv run ruff check .` clean
- [ ] `grep -rn "print(" src/` returns nothing
- [ ] Killing the model load path (e.g. bogus EMBED_MODEL) yields
      `index_status: "fallback_tfidf"` and non-empty TF-IDF results,
      not silent empty results
- [ ] Chunk JSONs contain `embedding_model` after a compress
- [ ] README updated: search section describes backend selection and
      fallback behavior accurately. No claims the code does not implement.
