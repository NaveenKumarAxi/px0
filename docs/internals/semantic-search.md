# Semantic Method Search Internals

Semantic search is opt-in (`semantic.enabled`, default false) and has three stages:

1. `collectSemanticMethods` walks the already-built `Index.Files()` list and applies `Outline` to supported source files. Method/function bodies are approximated from symbol lines and indentation and capped before they are sent anywhere.
2. `describeSemanticBatch` invokes the selected harness with a prompt containing up to four snippets at a time. The harness runs with a temporary directory as its working directory and must return a JSON array of `{id, description}` objects. `readOnlyAnalysisArgs` replaces unattended-edit modes with analysis/read-only modes for supported CLIs; Goose is refused because px0 cannot enforce read-only execution for it. The harness is used for language-model analysis, not embeddings.
3. `openAICompatibleEmbedder` sends descriptions (and later queries) to the configured embeddings endpoint. Vectors are added to a `github.com/coder/hnsw` in-memory graph; a metadata map holds each method's path, line, signature, and summary.

`GET /api/semantic/status` exposes enabled/indexing/ready/error state and progress. `GET /api/semantic/search?q=...` embeds a query, searches the HNSW graph with cosine distance, and returns navigable method hits. The UI's Semantic toggle uses these endpoints; normal `/api/search` is unchanged.

Embedding configuration is read from `PX0_EMBEDDING_BASE_URL`, `PX0_EMBEDDING_MODEL`, and `PX0_EMBEDDING_API_KEY`. A configured base URL can be any OpenAI-compatible embedding API; the base URL is suffixed with `/embeddings` unless it already ends there. If no base URL is set but an API key is present, the OpenAI `/v1` base URL is used. The API key is not stored in px0 settings.

The graph is process-local and rebuilt at initial workspace load when enabled, on explicit reindex, and when the selected harness changes. There is no persisted vector cache or incremental update path yet. Source parsing uses the existing regex outline rather than AST parsers, so results depend on the supported patterns in `symbols.go`.
