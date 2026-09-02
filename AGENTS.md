# Contributor guidance

This repository is an agent-development teaching project, not a generic conversion
library with a chat wrapper.

- Keep the model/tool/observation loop explicit and readable.
- Keep scientific parsing, writing, and validation deterministic.
- Never let an LLM invent coordinates, parameters, units, or semantic mappings.
- Every filesystem tool must use the shared `Workspace` boundary.
- New write tools must protect existing files by default.
- Add independent tool tests and mock-LLM orchestration tests for new behavior.
- Prefer Python 3.11+ type hints and small standard-library abstractions.
- Do not introduce an agent framework without an explicit architectural decision.
- Treat exact conversion, lossy conversion, semantic translation, and no-equivalent
  concepts as distinct outcomes.
