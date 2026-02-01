# Personal AI Assistant - Enhanced Codebase Documentation

This report provides a comprehensive overview of the Personal AI Assistant codebase, including the advanced features implemented to ensure a clean, organized, and context-aware experience.

## 1. File Structure

- `app/`: The core Chainlit-based chat application.
  - `app.py`: The main logic, orchestrating chat, tools, and background processing.
  - `tool_definitions.py`: JSON schemas for the active tools available to the AI.
  - `requirements.txt`: Python dependencies.
- `scheduler/`: Background service for long-term memory maintenance.
  - `scheduler.py`: Handles nightly recaps and vector database re-indexing.
- `vault/`: Human-readable Markdown storage for all knowledge.
  - `meta/`: System files including `instructions.md`, `learned_behaviors.md`, and `recent_context.md`.
  - `notes/`, `family/`, `recipes/`, etc.: Categorized knowledge.
- `compose.dev.yaml`: Docker configuration for the local development and testing environment.

## 2. Key Architectural Features

### Smart Merge (Wiki-Style Updates)
The system has moved away from "blind appending." When new information is learned or a file is updated:
1. The assistant **reads** the existing content.
2. An LLM **merges** the new facts into the existing structure, removing duplicates and refining existing data.
3. The file is **overwritten** with the clean, consolidated version.
This ensures that vault files remain concise and organized regardless of how many times a topic is discussed.

### Hybrid Active/Passive Agency
The assistant operates in two modes simultaneously:
- **Active Tools (Chat Agent)**: The user-facing AI can actively call tools (`list_files`, `read_file`, `update_file`, `search_files`) to find information or precisely modify the vault.
- **Passive Recording (Background Agent)**: Every exchange is monitored. If important facts are mentioned implicitly, they are automatically extracted and merged into the vault without user intervention.

### Live Context Awareness (Working Memory)
Unlike a static RAG system, this assistant maintains a "Live Log":
- **Immediate Logging**: As facts are extracted in the background, a summary (e.g., "[10:00] Learned new info about Lasagna") is appended to `meta/recent_context.md`.
- **Instant Persistence**: This ensures that even if a new chat session is started, the assistant is immediately aware of what was discussed earlier that day.

### Optimized UX
- **Lazy Loading**: The iCal calendar integration is processed in the background, allowing the chat interface to load instantly without waiting for network fetches.
- **Contextual Extraction**: Information extraction looks at the last 10 messages of history, allowing the AI to understand pronouns and references (e.g., "Save that to my schedule").

## 3. System Dependencies

- **Chainlit**: UI and session management.
- **ChromaDB**: Semantic search and long-term retrieval.
- **OpenRouter**: Access to advanced LLMs (Standardized on `google/gemini-3-flash-preview`).
- **Docker**: Containerization and service orchestration.

## 4. Workflow Summary

- **User sends a message.**

- **Assistant uses tools** to gather context if needed.

- **Assistant responds** with a streaming interface.

- **Background task** resolves the topic and extracts facts.

- **Smart Merge** integrates those facts into the Vault.

- **Recent Context** is updated to keep the assistant's working memory fresh.



## 5. Future Roadmap: MCP Integration



Research has been conducted into integrating the **Model Context Protocol (MCP)** using Chainlit's native features to replace or augment the current hard-coded tool system.



### Strategy

- **Chainlit Native Integration**: Leverage `@cl.on_mcp_connect` and `@cl.on_mcp_disconnect` handlers to manage external tool providers.

- **Standardized Tooling**: Move towards using MCP servers for common tasks (e.g., filesystem access, Google Drive, Spotify) to reduce custom Python maintenance.

- **Unified Tool Loop**: Update `app.py` to dynamically fetch tools from `cl.context.session.mcp_sessions` and present them to the LLM.



### Required Configuration

- **`chainlit.toml`**: Enable `features.mcp` and define `allowed_executables` (e.g., `npx`, `uvx`, `python`).

- **Subprocess Management**: Since the app runs in Docker, any MCP servers requiring specific runtimes (Node.js, Go) must have those runtimes installed in the `Dockerfile`.

- **Dynamic Discovery**: Implement logic in `on_mcp_connect` to list and store tools in the user session for LLM context injection.
