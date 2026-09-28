# Semantic Method Search

Semantic Method Search finds methods by describing intent rather than requiring an exact identifier or text match. In the workspace Search pane, enable **Semantic** and enter a natural-language query such as “where are configuration files parsed?”. Results show a method summary and jump to the method's source line. Turn Semantic off to return to normal literal/regex workspace search.

## Enable it

1. Install and select one of px0's supported coding harnesses in the Agent / AI settings or the first-use agent picker (Claude Code, Gemini CLI, Cursor Agent, Antigravity, OpenCode, Codex, Aider, or Goose).
2. Configure an OpenAI-compatible embeddings endpoint in the environment before starting px0:

   ```sh
   export PX0_EMBEDDING_API_KEY="your-provider-key"
   export PX0_EMBEDDING_BASE_URL="https://api.openai.com/v1"
   export PX0_EMBEDDING_MODEL="text-embedding-3-small"
   ```

   The base URL may point to another OpenAI-compatible embeddings service. If only the API key is set, the base URL defaults to OpenAI's `/v1` API. The model defaults to `text-embedding-3-small`.
3. In Settings, enable **Semantic Method Search**. It is off by default. On the next workspace load, px0 extracts methods, asks the selected harness for short descriptions in small batches, embeds those descriptions, and creates an in-memory HNSW index. The Search pane displays progress.

The selected harness/model is used only for method summaries; vector embeddings use the separately configured embedding endpoint, so the selected harness can be changed among supported models. px0 requests analysis/read-only modes for supported harnesses and runs them from a temporary working directory rather than from the project. Custom command templates must be configured with their own read-only mode. Goose currently cannot be used for semantic indexing because px0 cannot enforce a read-only run mode for its CLI. Changing the harness or using Reindex rebuilds the summaries and vectors.

## Privacy and limitations

Enabling the setting sends extracted method source to the selected coding harness and sends method summaries plus search queries to the configured embedding endpoint. Do not enable it for code that must not leave your machine unless both services are local or otherwise approved. px0 does not save the vector index to disk; it is rebuilt for each px0 process. Method extraction uses px0's existing lightweight, regex-based symbol outline, so language coverage and method boundaries are approximate. Reindex the workspace after changing files to refresh the semantic index.

Semantic Search requires the AI harness feature; `-no-agent` disables the harness and semantic indexing. Existing workspace text search works independently and remains local.
