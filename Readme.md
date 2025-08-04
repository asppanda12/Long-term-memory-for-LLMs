
# Long-Term Memory for LLMs - Setup Guide

This project provides a robust memory system for OpenAI GPT models, enabling them to persistently and efficiently remember information across conversations.

## What This System Does

- When a user says: "I use Shram and Magnet as productivity tools", the system stores this information as a memory.
- If the user later asks: "What are the productivity tools that I use?", the system responds: "You use Shram and Magnet".
- If the user says: "I don't use Magnet anymore", the system updates or deletes the relevant memory.
- The next time the user asks, the system responds: "You use Shram".

## Quick Start

1. Clone the repository and install dependencies:
    ```bash
    git clone <your-repo-url>
    cd llm-long-term-memory
    pip install -r requirements.txt
    ```

2. Set your OpenAI API key:
    ```bash
    echo "OPENAI_API_KEY=your_actual_openai_api_key_here" > .env
    ```

3. Test the system:
    ```bash
    python test_memory_system.py
    ```

4. Run the basic example:
    ```bash
    python llm_memory_system_openai.py
    ```

## Requirements

```
openai>=1.0.0
chromadb>=0.4.0
python-dotenv>=1.0.0
tiktoken>=0.5.0
pydantic>=2.0.0
numpy>=1.21.0
flask>=2.3.0
flask-cors>=4.0.0
```

## Core Usage

### Basic Memory-Enabled Chat

```python
from llm_memory_system_openai import MemoryEnabledGPT

gpt = MemoryEnabledGPT("your-openai-api-key")

# Store information
gpt.chat("I use Shram and Magnet as productivity tools", user_id="user123")

# Retrieve information
gpt.chat("What productivity tools do I use?", user_id="user123")
# Returns: "You use Shram and Magnet as productivity tools"

# Update information
gpt.chat("I don't use Magnet anymore", user_id="user123")

# Query updated memory
gpt.chat("What productivity tools do I use?", user_id="user123")
# Returns: "You use Shram for productivity"
```

### Memory-Only Queries

```python
memory_agent = LongTermMemoryAgent("your-openai-api-key")
answer = memory_agent.answer_with_memory("What tools do I use?", "user123")
```

## REST API Server

Start the API server:

```bash
python api_server.py
```

The server runs at `http://localhost:5000`.

### API Endpoints

- `/chat`: Chat with memory
- `/ask`: Query memory only
- `/user/<id>/profile`: Get user profile
- `/demo`: Run demo

Example usage:

```bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I use Shram and Magnet as productivity tools", "user_id": "user123"}'
```

## How It Works

1. **Memory Extraction**: Uses OpenAI GPT to analyze conversations and extract key facts, preferences, tools, goals, and experiences.
2. **Memory Storage**: Stores memories in a ChromaDB vector database, enabling semantic search and automatic updates.
3. **Memory Retrieval**: Finds relevant memories using semantic search and generates responses using stored context.
4. **Memory Management**: Supports adding, updating, and deleting memories based on user input.

## Configuration Options

Set environment variables in your `.env` file:

```
OPENAI_API_KEY=your-key
FLASK_ENV=development  # Optional: enable debug mode
PORT=5000              # Optional: API server port
```

You can also configure the memory agent:

```python
memory_agent = LongTermMemoryAgent(
    openai_api_key="your-key",
    db_path="./memory_db",
    collection_name="user_memories",
    embedding_model="text-embedding-3-small",
    chat_model="gpt-3.5-turbo"
)
```

## Memory Types

The system supports five types of memories:

| Type         | Description                | Example                              |
|--------------|---------------------------|--------------------------------------|
| tool_usage   | Software, apps, platforms | "Uses VS Code for development"       |
| preference   | Likes, dislikes, choices  | "Prefers dark mode interfaces"       |
| fact         | Personal information      | "Works as software engineer"         |
| goal         | Objectives, plans         | "Wants to learn machine learning"    |
| experience   | Past events, activities   | "Attended PyCon conference"          |

## Testing

Run all tests:

```bash
python test_memory_system.py
```

Test scenarios include memory storage, retrieval, updates, deletions, and persistence.

## Production Deployment

A Dockerfile and production instructions are provided. ChromaDB persists data in the `./memory_db/` directory.

## Security and Privacy

- All memories are stored locally by default.
- Each user's data is isolated and only accessible by their user ID.
- You can add authentication to the API for additional security.

## Advanced Configuration

- Customize memory extraction logic by subclassing `LongTermMemoryAgent`.
- Filter memories by type.
- Batch process multiple conversations.
- Monitor and analyze memory statistics.

## API Reference

### Core Classes

- `MemoryEnabledGPT`: Main class for memory-enabled conversations.
    - `chat(message, user_id, use_memory=True)`
    - `get_user_profile(user_id)`
    - `clear_user_memory(user_id)`

- `LongTermMemoryAgent`: Low-level memory management.
    - `extract_memories(conversation, user_id)`
    - `store_memory(memory)`
    - `search_similar_memories(query, user_id, top_k)`
    - `answer_with_memory(question, user_id)`

### REST Endpoints

| Method | Endpoint                | Description                  |
|--------|------------------------|------------------------------|
| POST   | /chat                  | Chat with memory             |
| POST   | /ask                   | Query memories only          |
| GET    | /user/<id>/profile     | Get user profile             |
| GET    | /user/<id>/memories    | Get user memories            |
| DELETE | /user/<id>/memories    | Delete user memories         |
| POST   | /search                | Search memories              |
| POST   | /memories/add          | Add memory manually          |
| GET    | /stats                 | System statistics            |
| GET    | /health                | Health check                 |
| GET    | /demo                  | Run demo conversation        |

## Success Criteria

- The system stores, retrieves, updates, and deletes memories as described.
- Memories persist across sessions.
- The implementation is efficient and ready for production.

---

This system is designed for efficient, accurate, and persistent memory management for LLMs. It is easy to set up, test, and deploy, and is ready for real-world use.