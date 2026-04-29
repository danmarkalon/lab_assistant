#!/usr/bin/env python3
"""
Generate method_support content for all methods with empty method_support docs.

Uses Gemini to synthesize expert-level content from:
1. Database files (protocols, calculation sheets, etc.)
2. Gemini's own knowledge of the technique
3. The Cell Fractionation method_support as a format template

Structure follows the Cell_fractionation template:
  Part 1: The Development Journey (lab record)
  Part 2: Protocol Reference (from database docs)
  Part 3: Expert System Prompt (for the bot persona)
"""
import sys, os, io, asyncio, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httplib2
httplib2.CA_CERTS = '/etc/ssl/certs/ca-certificates.crt'
os.environ['SSL_CERT_FILE'] = '/etc/ssl/certs/ca-certificates.crt'

# Patch certifi for genai client
import certifi
certifi.where = lambda: '/etc/ssl/certs/ca-certificates.crt'

from src.google_client import _get_service, append_doc_text
from src.config import DRIVE_ROOT_FOLDER_ID, GEMINI_API_KEY
from google import genai
from google.genai import types as genai_types
import docx

svc = _get_service('drive', 'v3')
ai = genai.Client(api_key=GEMINI_API_KEY)

def find_folder(parent_id, name):
    q = f"'{parent_id}' in parents and name='{name}' and trashed=false and mimeType='application/vnd.google-apps.folder'"
    res = svc.files().list(q=q, fields='files(id)').execute().get('files', [])
    return res[0]['id'] if res else None

def find_file(parent_id, name):
    q = f"'{parent_id}' in parents and name='{name}' and trashed=false"
    res = svc.files().list(q=q, fields='files(id,mimeType)').execute().get('files', [])
    return res[0] if res else None

def download_docx_text(file_id):
    content = svc.files().get_media(fileId=file_id).execute()
    doc = docx.Document(io.BytesIO(content))
    text = '\n'.join([p.text for p in doc.paragraphs if p.text.strip()])
    return text

def read_google_doc(doc_id):
    docs_svc = _get_service('docs', 'v1')
    doc = docs_svc.documents().get(documentId=doc_id).execute()
    text = ''
    for elem in doc.get('body', {}).get('content', []):
        if 'paragraph' in elem:
            for run in elem['paragraph'].get('elements', []):
                text += run.get('textRun', {}).get('content', '')
    return text.strip()

def list_all_readable(folder_id, prefix=''):
    """List all readable files recursively, returning (name, text) pairs for docs/docx."""
    q = f"'{folder_id}' in parents and trashed=false"
    items = svc.files().list(q=q, fields='files(id,name,mimeType)', orderBy='name', pageSize=100).execute().get('files', [])
    results = []
    for item in items:
        mime = item['mimeType']
        full_name = f"{prefix}{item['name']}"
        if 'folder' in mime:
            results.extend(list_all_readable(item['id'], f"{full_name}/"))
        elif mime == 'application/vnd.google-apps.document':
            try:
                text = read_google_doc(item['id'])
                if text.strip():
                    results.append((full_name, text))
            except Exception as e:
                print(f"  Warning: could not read Google Doc {full_name}: {e}")
        elif 'wordprocessingml' in mime:
            try:
                text = download_docx_text(item['id'])
                if text.strip():
                    results.append((full_name, text))
            except Exception as e:
                print(f"  Warning: could not read docx {full_name}: {e}")
        else:
            # For xlsx, pptx, csv, pdf - just record as available reference
            results.append((full_name, f"[Binary file: {mime.split('/')[-1]}]"))
    return results

def list_file_names(folder_id, prefix=''):
    """List all file names recursively."""
    q = f"'{folder_id}' in parents and trashed=false"
    items = svc.files().list(q=q, fields='files(id,name,mimeType)', orderBy='name', pageSize=100).execute().get('files', [])
    names = []
    for item in items:
        full_name = f"{prefix}{item['name']}"
        if 'folder' in item['mimeType']:
            names.extend(list_file_names(item['id'], f"{full_name}/"))
        else:
            names.append(full_name)
    return names

def generate_method_support(method_name, database_texts, file_list):
    """Use Gemini to generate method_support content."""
    # Build context from database docs
    db_context = ""
    for name, text in database_texts:
        if text.startswith('[Binary file:'):
            db_context += f"\n--- {name} (binary file, content not readable) ---\n"
        else:
            # Limit each doc to prevent token overflow
            truncated = text[:4000] if len(text) > 4000 else text
            db_context += f"\n--- {name} ---\n{truncated}\n"
    
    file_listing = "\n".join(f"  - {f}" for f in file_list)

    prompt = f"""You are an expert laboratory scientist creating a comprehensive "method_support" 
document for the method: "{method_name}".

This document will be loaded into an AI lab assistant's context to help it guide researchers 
through experiments using this method. The document has THREE parts:

**Part 1: The Development Journey (Lab Record)**
Write a narrative-style record that captures key decisions, trade-offs, and optimizations 
specific to this method. Include:
- Why specific reagents/approaches were chosen over alternatives
- Common pitfalls and how to avoid them
- Critical parameters that affect results
- Troubleshooting insights
Use your deep knowledge of this method combined with the database documents below.

**Part 2: Protocol Quick Reference**
Summarize the key protocols available in the database folder. List:
- Materials needed
- Key steps with critical parameters
- Stopping points
- Quality control checkpoints
Reference the specific protocol variants available in the database.

**Part 3: Expert System Prompt ("The [Method] Specialist")**
Create a system prompt persona for the AI assistant when helping with this method. Include:
- Role & expertise definition
- Core directives for guiding experiments
- A "Living Framework" of method-specific rules and checks
- Response structure guidelines

=== DATABASE FILES IN THIS METHOD'S FOLDER ===
{file_listing}

=== READABLE DATABASE DOCUMENT CONTENTS ===
{db_context}

=== FORMAT GUIDELINES ===
- Use clear section headers
- Include emoji markers for stopping points: [ 🛑 STOPPING POINT ]
- Include warning markers: (⚠️ ...)
- Use specific concentrations, temperatures, and times
- Be practical and bench-ready, not theoretical
- Reference specific files from the database when relevant

Generate the complete method_support document now. Be thorough and practical.
"""

    models = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash"]
    for model in models:
        for attempt in range(3):
            try:
                print(f"  Calling {model} for {method_name} (attempt {attempt+1})...")
                response = ai.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        temperature=0.7,
                        max_output_tokens=8000,
                    ),
                )
                return response.text
            except Exception as e:
                if '503' in str(e) or 'UNAVAILABLE' in str(e) or '429' in str(e):
                    wait = 15 * (attempt + 1)
                    print(f"  Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise
        print(f"  Model {model} exhausted, trying next...")
    raise RuntimeError(f"All models exhausted for {method_name}")


def write_to_google_doc(doc_id, content):
    """Write content to a Google Doc (append to empty doc)."""
    asyncio.run(append_doc_text(doc_id, content))
    print(f"  Written {len(content)} chars to doc {doc_id}")


# ── Method definitions ──────────────────────────────────────────────────────
METHODS = {
    'HCR FISH': {
        'method_support_id': '1im71-5E8T_wgTAWo0tZiJb_7UIQdQ2AhNgXpYPqiac4',
    },
    'stability': {
        'method_support_id': '1DDlkH7bq9cZ4C3qtFGpEyttNGxACZFsc9jfFQV3qf7I',
    },
}

def main():
    for method_name, info in METHODS.items():
        print(f"\n{'='*60}")
        print(f"Processing: {method_name}")
        print(f"{'='*60}")
        
        # Find method folder and database
        method_id = find_folder(DRIVE_ROOT_FOLDER_ID, method_name)
        if not method_id:
            print(f"  ERROR: Method folder not found!")
            continue
        
        db_id = find_folder(method_id, 'database')
        if not db_id:
            print(f"  ERROR: database/ folder not found!")
            continue
        
        # List all files
        file_list = list_file_names(db_id)
        print(f"  Found {len(file_list)} files in database/")
        
        # Read all readable documents
        db_texts = list_all_readable(db_id)
        readable_count = sum(1 for _, t in db_texts if not t.startswith('[Binary'))
        print(f"  Read {readable_count} text documents")
        
        # Generate method_support content
        content = generate_method_support(method_name, db_texts, file_list)
        print(f"  Generated {len(content)} chars of content")
        
        # Write to Google Doc
        doc_id = info['method_support_id']
        write_to_google_doc(doc_id, content)
        
        print(f"  ✓ {method_name} method_support filled!")
        
        # Rate limit between methods
        time.sleep(5)
    
    print(f"\n{'='*60}")
    print("All method_support documents filled!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
