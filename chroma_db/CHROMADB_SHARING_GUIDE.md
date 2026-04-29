# ChromaDB Knowledge Base - Import Guide

This guide explains how to integrate a pre-built ChromaDB vector database into your project. The database contains indexed OpenProject work packages with vector embeddings for semantic search.

---

## 📁 What You're Getting

The `chroma_db` folder contains:

```
chroma_db/
├── chroma.sqlite3              # Main SQLite database with vectors and metadata
├── sync_state.json             # Sync tracking (optional, can be deleted)
└── <uuid-folders>/             # Binary embedding data segments
```

**Collection Name:** `experiments`  
**Content:** OpenProject work packages with full-text descriptions, metadata, and vector embeddings

---

## 📦 Requirements

### Python Version
- Python 3.9 or higher

### Required Packages

Create a `requirements.txt` or install directly:

```txt
chromadb>=0.4.0
sentence-transformers>=2.2.0
```

Install with pip:

```bash
pip install chromadb>=0.4.0 sentence-transformers>=2.2.0
```

> **Note:** The first time you use sentence-transformers, it will download the embedding model (~420MB). Make sure you have internet access for the initial run.

---

## 🔧 Setup Instructions

### Step 1: Copy the Database

Copy the entire `chroma_db` folder to your project directory:

```bash
# Example: Copy to your project root
cp -r /path/to/shared/chroma_db ./chroma_db
```

**Important:** Copy the ENTIRE folder, including all UUID subfolders and the SQLite file. The structure must remain intact.

### Step 2: Verify the Structure

After copying, your project should look like:

```
your_project/
├── chroma_db/
│   ├── chroma.sqlite3
│   ├── sync_state.json          # Optional - can delete if not syncing
│   └── <uuid-folders>/
├── your_script.py
└── requirements.txt
```

---

## 💻 Usage Examples

### Basic Connection and Query

```python
import chromadb

# Connect to the persistent database
client = chromadb.PersistentClient(path="./chroma_db")

# Get the experiments collection
collection = client.get_collection(name="experiments")

# Check how many documents are in the database
print(f"Total documents: {collection.count()}")
```

### Semantic Search (Query by Meaning)

```python
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="experiments")

# Search for work packages by semantic meaning
results = collection.query(
    query_texts=["experiments with DNA sequencing"],
    n_results=5  # Return top 5 matches
)

# Print results
for i, (doc, metadata, distance) in enumerate(zip(
    results['documents'][0],
    results['metadatas'][0],
    results['distances'][0]
)):
    print(f"\n--- Result {i+1} (distance: {distance:.4f}) ---")
    print(f"ID: {metadata.get('id', 'N/A')}")
    print(f"Subject: {metadata.get('subject', 'N/A')}")
    print(f"Status: {metadata.get('status', 'N/A')}")
    print(f"Content Preview: {doc[:300]}...")
```

### Filter by Metadata

```python
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="experiments")

# Search with metadata filters
results = collection.query(
    query_texts=["recent experiments"],
    n_results=10,
    where={"status": "In progress"}  # Only return in-progress items
)

# Or filter by multiple conditions
results = collection.query(
    query_texts=["analysis results"],
    n_results=5,
    where={
        "$and": [
            {"status": {"$ne": "Closed"}},
            {"type": "Task"}
        ]
    }
)
```

### Get All Documents

```python
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="experiments")

# Get all documents (be careful with large databases)
all_docs = collection.get(
    include=["documents", "metadatas"]
)

print(f"Total documents: {len(all_docs['ids'])}")

# Iterate through all
for doc_id, metadata in zip(all_docs['ids'], all_docs['metadatas']):
    print(f"ID: {doc_id}, Subject: {metadata.get('subject', 'N/A')}")
```

### Get Specific Document by ID

```python
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="experiments")

# Get specific work package by its ID
result = collection.get(
    ids=["wp_547"],  # Work package IDs are prefixed with 'wp_'
    include=["documents", "metadatas"]
)

if result['documents']:
    print(result['documents'][0])
    print(result['metadatas'][0])
```

---

## 📊 Available Metadata Fields

Each document in the collection has these metadata fields:

| Field | Type | Description |
|-------|------|-------------|
| `id` | int | Work package ID number |
| `subject` | string | Work package title |
| `type` | string | Type (e.g., "Task", "Feature", "Bug") |
| `status` | string | Current status |
| `priority` | string | Priority level |
| `assignee` | string | Assigned person |
| `project` | string | Project name |
| `done_ratio` | int | Completion percentage (0-100) |
| `start_date` | string | Start date (ISO format or null) |
| `due_date` | string | Due date (ISO format or null) |
| `created_at` | string | Creation timestamp |
| `updated_at` | string | Last update timestamp |
| `experiment_id` | string | Extracted experiment number (e.g., "654") |

---

## 🔍 Complete Example Script

Save this as `query_knowledge_base.py`:

```python
#!/usr/bin/env python3
"""Query the OpenProject knowledge base using semantic search."""

import chromadb
import sys


def main():
    # Connect to database
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_collection(name="experiments")
    except Exception as e:
        print(f"Error connecting to database: {e}")
        print("Make sure the chroma_db folder exists and is properly copied.")
        sys.exit(1)
    
    print(f"Connected to knowledge base with {collection.count()} documents\n")
    
    # Interactive query loop
    while True:
        query = input("\nEnter your search query (or 'quit' to exit): ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        if not query:
            continue
        
        # Perform semantic search
        results = collection.query(
            query_texts=[query],
            n_results=5
        )
        
        if not results['documents'][0]:
            print("No results found.")
            continue
        
        print(f"\n{'='*60}")
        print(f"Top {len(results['documents'][0])} results for: '{query}'")
        print('='*60)
        
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ), 1):
            print(f"\n--- Result {i} ---")
            print(f"Work Package #{metadata.get('id', 'N/A')}: {metadata.get('subject', 'N/A')}")
            print(f"Status: {metadata.get('status', 'N/A')} | Type: {metadata.get('type', 'N/A')}")
            print(f"Relevance Score: {1 - distance:.2%}")
            print(f"\nDescription Preview:")
            print(doc[:500] + "..." if len(doc) > 500 else doc)


if __name__ == "__main__":
    main()
```

Run it:

```bash
python query_knowledge_base.py
```

---

## ⚠️ Troubleshooting

### "Collection not found" Error

```python
# List all available collections
client = chromadb.PersistentClient(path="./chroma_db")
print(client.list_collections())
```

The collection should be named `experiments`. If it shows a different name, use that name instead.

### "Database is locked" Error

Make sure no other process is using the database. ChromaDB doesn't support multiple writers.

### Slow First Query

The first query may be slow as sentence-transformers loads the embedding model. Subsequent queries will be faster.

### Missing UUID Folders

If you only copied `chroma.sqlite3`, the database won't work. You need ALL files and folders from `chroma_db/`.

---

## 🔗 Integration with LangChain (Optional)

If you want to use this with LangChain:

```bash
pip install langchain langchain-community
```

```python
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Use the same embeddings model ChromaDB uses internally
embeddings = HuggingFaceEmbeddings()

# Connect to existing database
vectorstore = Chroma(
    persist_directory="./chroma_db",
    collection_name="experiments",
    embedding_function=embeddings
)

# Use as a retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
docs = retriever.invoke("your search query")
```

---

## 📝 Notes

- The database is **read-only safe** - you can query without risk of corruption
- To add new documents, you'll need the original sync code from the source project
- The `sync_state.json` file tracks which work packages have been synced - you can safely delete it if you're only reading
- Vector similarity uses cosine distance (lower = more similar)

---

**Questions?** Contact the person who shared this database with you for project-specific questions about the indexed content.
