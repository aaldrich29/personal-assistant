# Personal AI Assistant

A self-hosted, private AI assistant that learns from your conversations and manages your daily life using a Retrieval-Augmented Generation (RAG) architecture.

> **⚠️ SECURITY WARNING**: This application has **NO AUTHENTICATION**. It is designed to run on a private network (localhost) or behind a secured reverse proxy (like Cloudflare Zero Trust, Tailscale, or Nginx with Basic Auth). Do NOT expose this directly to the public internet.

## Features

*   **🧠 Long-term Memory**: Uses a local Markdown "Vault" and ChromaDB vector search to remember facts, preferences, and family details forever.
*   **🗣️ Natural Conversation**: Built on Chainlit, supporting fluid chat and file uploads.
*   **📅 Calendar Integration**: Reads your iCal feed to brief you on your day.
*   **🔄 Self-Learning**: Automatically extracts and saves new information (preferences, events, health notes) to its vault in the background.
*   **📝 Obsidian Sync**: The vault is just a folder of Markdown files. Sync it with Obsidian (via WebDAV) to view and edit your assistant's brain directly.
*   **🔒 Privacy First**: All data is stored locally. The only external traffic is to the LLM provider (OpenRouter).

## Capabilities & Examples

### 🧠 It Knows You (Context Injection)
Unlike standard chatbots that start blank, this assistant **pre-loads context** before every conversation. It reads your `learned_behaviors.md`, `recent_context.md`, `daily_prep.md`, and checks your calendar.
*   **Result**: It knows it's your wife's birthday coming up or that you prefer concise answers without you having to remind it.

### 🔍 Total Recall (Semantic Search)
It indexes every file in your vault (Markdown). When you ask a question, it searches your entire history—not just the current chat window—to find the answer.
*   **Example**: "What did the doctor say about [Child]'s medication last month?"

### 🔄 It Learns & Adapts
You don't need to manually update files. If you mention "We're switching the dog to [Brand] food," the assistant extracts that fact and updates `family/pets/dog.md` and `learned_behaviors.md` automatically.

### 🌙 Nightly Review
A background scheduler runs every night to keep the brain organized:
1.  **Summarizes** the day's conversations into a log.
2.  **Updates** "Recent Context" so it remembers what's current.
3.  **Preps** a "Daily Prep" file for the next morning (reminders, incomplete tasks).
4.  **Re-indexes** the database to ensure search is fast and accurate.

### Example Queries
*   "Draft a grocery list for this week's meals, but remember we're out of town Friday."
*   "What size shoe does [Child] wear now?" (If you mentioned it previously)
*   "Summarize the vacation plans we talked about last week."
*   "Add a reminder to call the school on Monday morning."

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
    *   **Recommended**: Change the `WEBDAV_PASS` to secure your vault.
    *   **Optional**: Add your `ICAL_URL` if you want calendar integration.

### 2. Run

```bash
docker compose up -d
```

Docker will pull the necessary images, create your local `vault/` directory, and start the services.

Access the services:
*   **Chat Interface**: [http://localhost:8501](http://localhost:8501)
*   **WebDAV Server**: [http://localhost:8080](http://localhost:8080) (Use the user/pass from your `.env`)

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
