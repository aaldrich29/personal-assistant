import os

# Fix ONNX runtime thread affinity errors in Docker - must be set before importing onnxruntime
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["ORT_DISABLE_CPU_AFFINITY"] = "1"

# Suppress ONNX runtime thread affinity errors (they're harmless but noisy in Docker)
import onnxruntime as ort
ort.set_default_logger_severity(4)  # 4 = fatal only, suppresses the pthread_setaffinity_np errors

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from openai import OpenAI
import chromadb

# ============================================================================
# Configuration
# ============================================================================

VAULT_PATH = Path(os.environ.get("VAULT_PATH", "/vault"))
CHROMA_HOST = os.environ.get("CHROMA_HOST", "chromadb")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", 8000))
LLM_MODEL = os.environ.get("LLM_MODEL", "google/gemini-2.0-flash-001")

client = OpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# ============================================================================
# Vault Operations
# ============================================================================

def read_vault_file(relative_path: str) -> str:
    file_path = VAULT_PATH / relative_path
    if file_path.exists():
        return file_path.read_text()
    return ""

def write_vault_file(relative_path: str, content: str):
    file_path = VAULT_PATH / relative_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content)

def append_vault_file(relative_path: str, content: str):
    file_path = VAULT_PATH / relative_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "a") as f:
        f.write(content)

def ensure_vault_setup():
    """Ensure essential vault files exist with default content."""
    defaults = {
        "meta/instructions.md": """# Personal Assistant Instructions

You are a personal AI assistant for a busy family. Your role is to be genuinely helpful, remember important details, and make daily life easier.

## Your Personality
- Warm and friendly, like a trusted friend
- Concise but thorough - respect their time
- Proactive when helpful, but not pushy
- Remember you're talking to a real person managing a busy life

## What You Help With

### Family Management
- Remember each family member's name, age, activities, and preferences
- Track kids' school activities, homework, accomplishments
- Note important milestones and events

### Health & Wellness
- Track health information, allergies, medications
- Remember doctor appointments and notes
- Monitor sleep patterns, symptoms, or concerns mentioned
- Support workout routines and fitness goals

### Daily Life
- Help with meal planning based on preferences and dietary needs
- Remember favorite recipes and food preferences
- Track schedules and routines
- Generate to-do lists based on what you know

### Learning & Adapting
- Notice patterns in how the family operates
- Learn preferences without being told explicitly
- Adapt your suggestions based on past conversations
- Remember how they like things done

## How to Learn

When you hear important information:
1. Naturally acknowledge it in conversation
2. It will be automatically saved to your knowledge vault
3. Use it in future conversations when relevant

## Important Guidelines
- Never share family information externally
- Treat health information with extra care
- When unsure, ask clarifying questions
- If something seems urgent or concerning, acknowledge it appropriately
- Be helpful with curriculum-based questions using stored school information

## Daily Rhythm
- In the morning, be ready with relevant reminders
- During the day, help with whatever comes up
- Remember that context from earlier conversations carries forward
""",
        "meta/learned_behaviors.md": """# Learned Behaviors

This file is automatically updated by the assistant as it learns patterns and preferences.

## Communication Preferences
*How the family prefers to receive information*

## Routine Patterns
*Regular schedules and habits the assistant has noticed*

## Follow-up Items
*Things to check back on or ask about*

## Custom Rules
*Specific rules the assistant has learned to follow*
""",
        "meta/recent_context.md": "# Recent Context\n\n*Summary of recent conversations and events*\n",
        "meta/daily_prep.md": "# Daily Prep\n\n*Reminders and context for the upcoming day*\n",
        "meta/preferences.md": "# User Preferences\n\n*Preferences will be learned automatically from conversations.*\n",
    }
    for path, content in defaults.items():
        file_path = VAULT_PATH / path
        if not file_path.exists():
            print(f"[INIT] Creating default vault file: {path}")
            write_vault_file(path, content)

def get_todays_conversations() -> str:
    """Get all conversations from today."""
    today = datetime.now().strftime("%Y-%m-%d")
    conv_dir = VAULT_PATH / "conversations" / today

    if not conv_dir.exists():
        return ""

    all_conversations = []
    for conv_file in sorted(conv_dir.glob("*.md")):
        all_conversations.append(conv_file.read_text())

    return "\n\n---\n\n".join(all_conversations)

# ============================================================================
# Nightly Recap
# ============================================================================

def nightly_recap():
    """Generate nightly recap and update context."""
    print(f"[{datetime.now()}] Starting nightly recap...")

    # Get today's conversations
    conversations = get_todays_conversations()
    if not conversations:
        print("No conversations to recap")
        update_daily_prep("")
        return

    # Load current family info for context
    family_context = ""
    family_dir = VAULT_PATH / "family" / "members"
    if family_dir.exists():
        for member_file in family_dir.glob("*.md"):
            family_context += f"\n{member_file.read_text()}\n"

    # Load current learned behaviors
    learned_behaviors = read_vault_file("meta/learned_behaviors.md")

    # Load prompt template
    prompt_path = "meta/prompts/nightly_recap.md"
    recap_prompt_template = read_vault_file(prompt_path)
    
    # Fallback if file doesn't exist
    if not recap_prompt_template:
        print(f"Warning: Prompt file {prompt_path} not found. Using default.")
        recap_prompt_template = """Analyze today's conversations and create a structured summary.

Current Family Information:
{family_context}

Current Learned Behaviors:
{learned_behaviors}

Today's Conversations:
{conversations}

Create a response with these exact sections:

## Daily Summary
A brief 2-3 sentence summary of what was discussed today.

## Key Information Learned
Bullet points of important facts learned about:
- Family members (activities, accomplishments, concerns)
- Health information
- Schedule/appointment mentions
- Preferences or habits discovered
- Any other notable information

## Tomorrow's Reminders
Any items mentioned that are relevant for tomorrow or upcoming days.

## Patterns Noticed
Any patterns in behavior, preferences, or routines that the assistant should remember going forward.

## Behavior Updates
NEW patterns, preferences, or follow-up items that should be added to the learned behaviors file. Format as:
- communication_preferences: [any new communication preferences discovered]
- routine_patterns: [any new routine patterns noticed]
- follow_up_items: [things to follow up on with the user]
- custom_rules: [any new rules or preferences for how to handle things]

Only include items that are NEW and not already in the current learned behaviors.

Keep it concise but comprehensive. Focus on actionable information."""

    recap_prompt = recap_prompt_template.format(
        family_context=family_context if family_context else "No family members recorded yet.",
        learned_behaviors=learned_behaviors if learned_behaviors else "No learned behaviors yet.",
        conversations=conversations
    )

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are analyzing conversations to extract and organize important family information. Be concise and factual."},
                {"role": "user", "content": recap_prompt}
            ],
            temperature=0.3,
            max_tokens=2500,
            extra_body={
                "reasoning": {
                    "effort": "medium"
                }
            }
        )

        recap_content = response.choices[0].message.content

        # Save daily recap
        today = datetime.now().strftime("%Y-%m-%d")
        day_of_week = datetime.now().strftime("%A")
        write_vault_file(
            f"daily/{today}.md",
            f"# Daily Recap - {day_of_week}, {today}\n\n{recap_content}"
        )

        # Update recent context
        update_recent_context(today, recap_content)

        # Update daily prep for tomorrow
        update_daily_prep(recap_content)

        # Update learned behaviors
        update_learned_behaviors_from_recap(recap_content)

        print(f"[{datetime.now()}] Nightly recap completed")

    except Exception as e:
        print(f"Error in nightly recap: {e}")

def update_recent_context(date: str, recap_content: str):
    """Update the recent context file with latest information."""
    existing = read_vault_file("meta/recent_context.md")

    # Extract key information section
    key_info = ""
    if "## Key Information Learned" in recap_content:
        start = recap_content.find("## Key Information Learned")
        end = recap_content.find("## Tomorrow's Reminders")
        if end == -1:
            end = recap_content.find("## Patterns Noticed")
        if end == -1:
            end = len(recap_content)
        key_info = recap_content[start:end].strip()

    new_entry = f"\n\n### {date}\n{key_info}\n"

    if existing:
        if "# Recent Context" in existing:
            header_end = existing.find("\n", existing.find("# Recent Context"))
            header = existing[:header_end]
            rest = existing[header_end:]
        else:
            header = "# Recent Context\n\nRecent important information from conversations."
            rest = existing

        new_context = f"{header}\n{new_entry}{rest}"
    else:
        new_context = f"# Recent Context\n\nRecent important information from conversations.\n{new_entry}"

    lines = new_context.split("\n")
    if len(lines) > 300:
        new_context = "\n".join(lines[:300])

    write_vault_file("meta/recent_context.md", new_context)

def update_daily_prep(recap_content: str):
    """Update tomorrow's daily prep."""
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    tomorrow_day = (datetime.now() + timedelta(days=1)).strftime("%A")

    # Extract tomorrow's reminders
    reminders = ""
    if "## Tomorrow's Reminders" in recap_content:
        start = recap_content.find("## Tomorrow's Reminders")
        end = recap_content.find("## Patterns Noticed")
        if end == -1:
            end = recap_content.find("## Behavior Updates")
        if end == -1:
            end = len(recap_content)
        reminders = recap_content[start:end].strip()

    # Extract follow-up items from learned behaviors
    learned = read_vault_file("meta/learned_behaviors.md")
    follow_ups = ""
    if "## Follow-up Items" in learned:
        start = learned.find("## Follow-up Items")
        end = learned.find("\n## ", start + 1)
        if end == -1:
            end = len(learned)
        follow_section = learned[start:end]
        # Get recent follow-up items (last 5)
        lines = [l for l in follow_section.split("\n") if l.strip().startswith("-")]
        if lines:
            follow_ups = "\n## Pending Follow-ups\n" + "\n".join(lines[-5:])

    default_reminders = "## Reminders for Today\nNo specific reminders from yesterday."
    prep_content = f"""# Daily Prep - {tomorrow_day}, {tomorrow}

{reminders if reminders else default_reminders}

{follow_ups}

## Notes
- Review any pending items from previous conversations
- Check if there are scheduled activities mentioned for today
"""

    write_vault_file("meta/daily_prep.md", prep_content)

def update_learned_behaviors_from_recap(recap_content: str):
    """Extract and update learned behaviors from nightly recap."""
    if "## Behavior Updates" not in recap_content:
        return

    try:
        # Extract the behavior updates section
        start = recap_content.find("## Behavior Updates")
        behavior_section = recap_content[start:]

        timestamp = datetime.now().strftime('%Y-%m-%d')

        # Parse each category
        categories = {
            "communication_preferences": "## Communication Preferences",
            "routine_patterns": "## Routine Patterns",
            "follow_up_items": "## Follow-up Items",
            "custom_rules": "## Custom Rules"
        }

        current_behaviors = read_vault_file("meta/learned_behaviors.md")

        for key, header in categories.items():
            # Look for this category in the recap
            pattern = f"- {key}:"
            if pattern in behavior_section.lower():
                # Find the content after this pattern
                idx = behavior_section.lower().find(pattern)
                line_end = behavior_section.find("\n", idx)
                if line_end == -1:
                    line_end = len(behavior_section)

                content = behavior_section[idx + len(pattern):line_end].strip()

                # Skip if empty or just brackets
                if content and content not in ["[]", "[none]", "[nothing new]", ""]:
                    # Clean up the content
                    content = content.strip("[]").strip()
                    if content:
                        new_entry = f"\n- [{timestamp}] {content}"

                        if header in current_behaviors:
                            # Append to existing section
                            parts = current_behaviors.split(header)
                            if len(parts) >= 2:
                                rest = parts[1]
                                next_section = rest.find("\n## ")
                                if next_section != -1:
                                    section_content = rest[:next_section]
                                    after_section = rest[next_section:]
                                    current_behaviors = parts[0] + header + section_content + new_entry + after_section
                                else:
                                    current_behaviors = parts[0] + header + rest + new_entry
                        else:
                            # Add new section
                            current_behaviors += f"\n\n{header}{new_entry}"

        write_vault_file("meta/learned_behaviors.md", current_behaviors)
        print(f"[{datetime.now()}] Updated learned behaviors")

    except Exception as e:
        print(f"Error updating learned behaviors: {e}")

# ============================================================================
# Vault Reindexing
# ============================================================================

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def reindex_vault():
    """Re-index the entire vault to ChromaDB."""
    print(f"[{datetime.now()}] Starting vault reindex...")

    try:
        chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        collection = chroma_client.get_or_create_collection(
            name="vault_documents",
            metadata={"hnsw:space": "cosine"}
        )

        md_files = list(VAULT_PATH.rglob("*.md"))
        indexed = 0

        for file_path in md_files:
            try:
                content = file_path.read_text()
                if not content.strip():
                    continue

                relative_path = str(file_path.relative_to(VAULT_PATH))
                parts = file_path.relative_to(VAULT_PATH).parts
                category = parts[0] if parts else "general"

                chunks = chunk_text(content)

                for i, chunk in enumerate(chunks):
                    doc_id = f"{relative_path}_chunk_{i}"
                    collection.upsert(
                        ids=[doc_id],
                        documents=[chunk],
                        metadatas=[{
                            "source": relative_path,
                            "category": category,
                            "chunk_index": i,
                            "total_chunks": len(chunks)
                        }]
                    )
                    indexed += 1

            except Exception as e:
                print(f"Error indexing {file_path}: {e}")

        print(f"[{datetime.now()}] Reindexed {indexed} chunks from {len(md_files)} files")

    except Exception as e:
        print(f"Error in reindex: {e}")

# ============================================================================
# Scheduler Setup
# ============================================================================

scheduler = BlockingScheduler()

# Nightly recap at 11 PM
scheduler.add_job(
    nightly_recap,
    CronTrigger(hour=23, minute=0),
    id="nightly_recap",
    name="Nightly Recap",
    replace_existing=True
)

# Reindex vault at 11:30 PM
scheduler.add_job(
    reindex_vault,
    CronTrigger(hour=23, minute=30),
    id="reindex_vault",
    name="Reindex Vault",
    replace_existing=True
)

# Startup reindex after delay
scheduler.add_job(
    reindex_vault,
    'date',
    run_date=datetime.now() + timedelta(seconds=30),
    id="startup_reindex",
    name="Startup Reindex"
)

if __name__ == "__main__":
    # Ensure vault is ready
    ensure_vault_setup()
    
    print(f"[{datetime.now()}] Starting scheduler...")
    print("Scheduled jobs:")
    for job in scheduler.get_jobs():
        print(f"  - {job.name}: {job.trigger}")

    scheduler.start()
