# Personal AI Assistant

A self-hosted, private AI assistant that learns from your conversations and manages your daily life using a Retrieval-Augmented Generation (RAG) architecture.

## Features

*   **🧠 Long-term Memory**: Uses a local Markdown "Vault" and ChromaDB vector search to remember facts, preferences, and family details forever.
*   **🗣️ Natural Conversation**: Built on Chainlit, supporting fluid chat and file uploads.
*   **📂 Vault Editor**: Built-in web interface to view, edit, and manage your markdown notes and folder structure.
*   **📅 Calendar Integration**: Reads your iCal feed to brief you on your day.
*   **🌐 Web Search**: Can search the live web for news, weather, or facts using a delegated search tool.
*   **👁️ Vision Support**: Understands uploaded images, screenshots, and documents (PDFs).
*   **🔄 Self-Learning**: Automatically extracts and saves new information (preferences, events, health notes) to its vault in the background.
*   **🔒 Privacy First**: All data is stored locally. Unified authentication protects your chat, notes, and sync.

## Quick Start

### Prerequisites

1.  [Docker](https://docs.docker.com/get-docker/) installed.
2.  An API Key from [OpenRouter](https://openrouter.ai/).

### 1. Setup

You don't need to clone the full repository to run the assistant. You just need the configuration files.

1.  **Download Files**: Download `compose.yaml` and `.env.example` from this repository.
2.  **Configure Environment**:
    *   Rename `.env.example` to `.env`.
    *   Open `.env` in a text editor.
    *   **Required**: Paste your `OPENROUTER_API_KEY`.
    *   **Authentication**:
        *   Set `AUTH_USERNAME` and `AUTH_PASSWORD`. These protect all web interfaces and WebDAV.
        *   Set `CHAINLIT_AUTH_SECRET`. Generate a random string using `openssl rand -base64 32` or any password generator.
    *   **Models**: The assistant uses a split model approach:
        *   `LLM_MODEL`: Used for standard chat (Default: `google/gemini-3-flash-preview`).
        *   `SEARCH_MODEL`: Used for online web searches (Default: `:online` variant).
    *   **Optional**: Add your `ICAL_URL` and set your `TZ` (Timezone, e.g., `America/Chicago`).

### 2. Run

```bash
docker compose up -d
```

Access the services (using your `AUTH_USERNAME` and `AUTH_PASSWORD`):
*   **Chat Interface**: [http://localhost:8501](http://localhost:8501)
*   **Vault Editor**: [http://localhost:8081](http://localhost:8081)
*   **WebDAV Server**: [http://localhost:8080](http://localhost:8080)

**Developer Mode**
If you want to modify the source code or build the images yourself:
1.  Clone the repository: `git clone https://github.com/aaldrich29/personal-assistant.git`
2.  Use the dev compose file: `docker compose -f compose.dev.yaml up -d --build`
*   **WebDAV Server**: [http://localhost:8080](http://localhost:8080) (User: `admin`, Pass: `changeme` - change in `.env`)

### 3. First Steps

1.  Open the chat at `http://localhost:8501`.
2.  Introduce yourself! "Hi, I'm [Name]. My wife is [Name] and we have 2 kids."
3.  The assistant will learn this and save it to `vault/family/members/`.
4.  Check `vault/meta/learned_behaviors.md` later to see what it learned.

## Syncing with Obsidian (Optional)

To view and edit the assistant's memory directly in Obsidian:

1.  **Create a Vault**: Open a new, empty vault in Obsidian.
2.  **Install Plugin**: Install and enable the **Remotely Save** community plugin.
3.  **Configure WebDAV**:
    *   **Choose Service**: WebDAV
    *   **Address**: `http://localhost:8080/`
    *   **Username/Password**: Use the values from your `.env`.
    *   **Remote Base Directory** (or Root Folder Override): Enter `Assistant`.
4.  **Sync**: Click the sync button. The `family`, `notes`, and `meta` folders will populate your vault.

## Architecture

*   **App (`chainlit`)**: The main Python application handling the UI and RAG logic.
*   **Scheduler**: A background service that runs nightly tasks (summarizing the day's chat, re-indexing the vector database).
*   **ChromaDB**: A local vector database that stores embeddings of your vault files for semantic search.
*   **WebDAV**: A simple server to allow external access to the `vault` directory.

## Customization

### The Vault
The `vault/` directory is the source of truth. You can manually edit any file here.
*   `vault/meta/instructions.md`: Change the system prompt and personality.
*   `vault/family/members/`: Add/edit family member profiles.

### Docker Images
This repository includes a GitHub Workflow to build and push images. If you fork this, you can enable GitHub Actions to publish your own images to GHCR, allowing you to deploy on other servers using just the `compose.yaml` file without building locally.
