# AGENTS.md

FastRag is a Python 3.13 hybrid retrieval library and CLI: BM25 + dense vectors, RRF/rerank, and a LangGraph planning/answering flow. README, docstrings, comments, prompts, and CLI output are Chinese; keep new user-facing text and docs in Chinese.

## Setup and commands

- `uv sync`, then verify with `uv run fastrag --help` (`uv run python -m fastrag --help` is the same entry).
- Full setup: copy `.env.example` to `.env` and fill the endpoint key, then `uv run fastrag download` -> `uv run fastrag prepare`.
- `prepare` = chunk -> embed -> index. It re-embeds only new/changed chunks, but a fresh corpus still means minutes and API cost; don't run it unless artifacts are missing or the corpus changed.
- Inspect/debug: `uv run fastrag search "..." -k 5`, `rerank "..." -n 5`, `rrf "..." -n 5`, `embed --query "..."`, `ask`.
- All-local profile: `uv run fastrag --config local prepare` / `ask`. It expects llama.cpp on 9931 (embeddings) and 9938 (LLM), stores vectors in `data/embeddings-bge-m3`, and needs `NO_PROXY=127.0.0.1,localhost` behind a SOCKS proxy.
- Global flags `--config`, `-v`, `--version` may appear before or after the subcommand.
- There is no test suite, CI, linter, or formatter config. Verify behavior with the relevant CLI command; don't assume `pytest` or `ruff` exists.

## Architecture

- Dependency direction is one-way: `cli -> pipeline -> index/retrieve/llm -> ingest/config -> domain/errors`. Keep new code on that path. `domain.py` holds the shared immutable types (`Chunk`, `ScoredChunk`, `AnswerResult`, `AnswerEvent`) and the `SearchPlan` pydantic model.
- `cli/main.py` owns all argparse and returns an exit code; command modules in `cli/commands/` only provide `add_parser` + `run(args, config)`. Register new commands in `cli/commands/__init__.py`.
- `RagPipeline` (`pipeline/runner.py`) wraps the compiled graph. `ResourceBundle` lazily loads the store, vectors, BM25 index, reranker, and LLM chains, so `import fastrag` and pipeline construction do no I/O. `ask()` / `stream()` load `.env`; `RagPipeline.from_config()` does not.
- Graph flow: `plan_search` -> parallel `retrieve_bm25` / `retrieve_vector` -> `fuse` (RRF or rerank) -> `answer`. Rerank failure degrades to RRF; route failure falls back to the raw question; LLM retries use the SDK's `max_retries`, then `[llm.*.fallback]`; empty answers also trigger fallback.
- Retrieval and fusion exchange `ScoredChunk`, not raw dicts or LangChain `Document`s. Channels are named (`bm25`, `vector`), and `fusion.weights` is a dict keyed by channel name.

## Config, prompts, endpoints

- There is one TOML. Pydantic uses `extra="forbid"`, so typos and invalid values fail at load. Add fields in `config/models.py`; resolution and `.env` loading live in `config/loader.py`.
- Resolution: `--config` path/name > `FASTRAG_CONFIG` > `$FASTRAG_HOME/config/fastrag.toml` > packaged `src/fastrag/resources/config/fastrag.toml`; `--config default` means the normal chain. `FASTRAG_HOME` defaults to the repo root (editable) or current directory (wheel); relative paths and `.env` resolve under it.
- `load_config()` never reads `.env`; call `load_env()` explicitly when needed. The CLI and `fastrag.ask/stream` already do.
- Prompts live in `src/fastrag/resources/prompts/` and can be overridden by `$FASTRAG_HOME/prompts/`. They use `string.Template` `$name` placeholders, not `str.format`; unknown placeholders are left intact.
- Endpoints are OpenAI-compatible. Keep `provider = "openai"` for them; an empty `api_key_env` means a local service (placeholder key `local`). `thinking = false` adds `enable_thinking = false`, and `extra_body` is merged into the request.
- `retrieval.top_n` is the CLI default and ranking cutoff; optional `retrieval.context_chunks` controls how many chunks enter the answer prompt.

## Data artifacts

- `data/chunks.jsonl` is the single source of truth. Chunk ids are content-addressed (`sha1(source + "\0" + text)[:16]`), `source` is relative to `FASTRAG_HOME`, and `text == original[start:end]`.
- `ChunkStore.load` computes the full-file sha256. Embedding and BM25 manifests (`manifest.json` / `meta.json`) use `ArtifactKey(kind, source_digest, params)`.
- Vectors are normalized float32 and reused per chunk id; only new/changed ids are re-embedded, and a model/dimension change forces a full rebuild. BM25 stores only ids and resolves through `ChunkStore`; it raises if the digest or `[bm25]` params changed. All artifacts are written atomically (temp file + rename).
- Prefer `fastrag prepare`. After a corpus change vectors are incremental, but BM25 must be rebuilt. `data/`, `dist/`, and `.env` are gitignored.
- BM25 `stopwords` accepts a supported language name or a list (`[]` disables); `min_score` filters with a strict `score > min_score`.

## Conventions

- `.gitattributes` enforces LF; don't introduce CRLF.
- `opencode.json` configures the LangChain docs/reference MCP servers; use them for API questions.
- `bot-reviews/` is historical review material, not current instructions.
