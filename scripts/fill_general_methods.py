#!/usr/bin/env python3
"""
Read all readable files from the 'general protocols' folder and generate
a general_methods_assistant Google Doc.
"""
import sys, os, io, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httplib2
httplib2.CA_CERTS = '/etc/ssl/certs/ca-certificates.crt'
os.environ['SSL_CERT_FILE'] = '/etc/ssl/certs/ca-certificates.crt'
import certifi
certifi.where = lambda: '/etc/ssl/certs/ca-certificates.crt'

from src.google_client import _get_service, append_doc_text
from src.config import DRIVE_ROOT_FOLDER_ID, GEMINI_API_KEY
from google import genai
from google.genai import types as genai_types
import docx
import asyncio

svc = _get_service('drive', 'v3')
ai = genai.Client(api_key=GEMINI_API_KEY)


def find_folder(parent_id, name):
    q = f"'{parent_id}' in parents and name='{name}' and trashed=false and mimeType='application/vnd.google-apps.folder'"
    res = svc.files().list(q=q, fields='files(id)').execute().get('files', [])
    return res[0]['id'] if res else None


def download_docx_text(file_id):
    content = svc.files().get_media(fileId=file_id).execute()
    doc = docx.Document(io.BytesIO(content))
    text = '\n'.join([p.text for p in doc.paragraphs if p.text.strip()])
    return text


def download_doc_text(file_id):
    """Download old .doc format by exporting as plain text."""
    content = svc.files().get_media(fileId=file_id).execute()
    # .doc files - try reading as bytes
    try:
        return content.decode('utf-8', errors='replace')
    except:
        return content.decode('latin-1', errors='replace')


def read_google_doc(doc_id):
    docs_svc = _get_service('docs', 'v1')
    doc = docs_svc.documents().get(documentId=doc_id).execute()
    text = ''
    for elem in doc.get('body', {}).get('content', []):
        if 'paragraph' in elem:
            for run in elem['paragraph'].get('elements', []):
                text += run.get('textRun', {}).get('content', '')
    return text.strip()


def read_all_files_recursive(folder_id, prefix=''):
    """Read all files recursively, return list of (path, text_or_tag)."""
    q = f"'{folder_id}' in parents and trashed=false"
    items = svc.files().list(
        q=q, fields='files(id,name,mimeType)', orderBy='name', pageSize=100
    ).execute().get('files', [])
    results = []
    for item in items:
        mime = item['mimeType']
        full = f"{prefix}{item['name']}"
        if 'folder' in mime:
            results.extend(read_all_files_recursive(item['id'], f"{full}/"))
        elif mime == 'application/vnd.google-apps.document':
            try:
                text = read_google_doc(item['id'])
                results.append((full, text if text.strip() else '[empty]'))
            except Exception as e:
                results.append((full, f'[error: {e}]'))
        elif 'wordprocessingml.document' in mime:  # .docx
            try:
                text = download_docx_text(item['id'])
                results.append((full, text if text.strip() else '[empty]'))
            except Exception as e:
                results.append((full, f'[error: {e}]'))
        elif mime == 'application/msword':  # .doc
            try:
                text = download_doc_text(item['id'])
                # .doc binary produces mostly garbage; extract what looks like text
                import re
                # Filter to lines with mostly printable ASCII
                lines = text.split('\n')
                clean_lines = []
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    printable = sum(1 for c in line if c.isprintable())
                    if len(line) > 5 and printable / len(line) > 0.8:
                        clean_lines.append(line)
                clean = '\n'.join(clean_lines[:100])  # Cap at 100 lines
                results.append((full, clean if clean.strip() else '[binary .doc - not parseable]'))
            except Exception as e:
                results.append((full, f'[error: {e}]'))
        elif mime == 'text/plain':
            try:
                content = svc.files().get_media(fileId=item['id']).execute()
                text = content.decode('utf-8', errors='replace')
                results.append((full, text))
            except Exception as e:
                results.append((full, f'[error: {e}]'))
        else:
            tag = mime.split('/')[-1]
            results.append((full, f'[binary: {tag}]'))
    return results


def generate_with_retry(prompt, method_name):
    models = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash"]
    for model in models:
        for attempt in range(3):
            try:
                print(f"  Calling {model} (attempt {attempt+1})...")
                response = ai.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        temperature=0.7,
                        max_output_tokens=12000,
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


def main():
    # 1. Find general protocols folder
    gp_id = find_folder(DRIVE_ROOT_FOLDER_ID, 'general protocols')
    if not gp_id:
        print("ERROR: 'general protocols' folder not found!")
        return

    # 2. Read all files
    print("Reading all files in 'general protocols'...")
    all_files = read_all_files_recursive(gp_id)
    print(f"Found {len(all_files)} items total")
    for name, text in all_files:
        tag = f"{len(text)} chars" if not text.startswith('[') else text
        print(f"  {name}: {tag}")

    # 3. Build context for Gemini
    db_context = ""
    file_listing = []
    for name, text in all_files:
        file_listing.append(name)
        if text.startswith('['):
            db_context += f"\n--- {name} ---\n{text}\n"
        else:
            truncated = text[:5000] if len(text) > 5000 else text
            db_context += f"\n--- {name} ---\n{truncated}\n"

    file_list_str = "\n".join(f"  - {f}" for f in file_listing)

    prompt = f"""You are an expert laboratory scientist creating a comprehensive 
"general_methods_assistant" document for a lab AI assistant.

This document will be loaded into EVERY session of the AI lab assistant, regardless 
of which specific method the researcher is working on. It provides shared bench skills 
and cross-method knowledge.

The lab has these SPECIFIC methods as separate folders: Bone Marrow FACS, Cell Fractionation, 
HCR FISH, and Stability assays. The "general protocols" folder contains shared skills 
used across ALL of these methods.

Generate a document with these sections:

**Section 1: Buffer Master Reference**
Summarize all common laboratory buffers, their compositions, and typical uses.
Draw from the "all buffers and use.xlsx" file listing and your own expert knowledge
of which buffers are used in the lab's methods.

**Section 2: BCA Protein Assay**
- Principle and when to use BCA vs Bradford
- Standard curve preparation (BSA standards)
- Sample preparation considerations
- Calculation method and expected ranges
- Common pitfalls (detergent interference, reducing agents)
- Reference the BCA.xlsx calculation sheet

**Section 3: Jess Western Blot (ProteinSimple)**
- System overview and advantages over traditional Western
- Antibody calibration workflow (reference the calibration plans/tables)
- Immunoassay + total protein protocol summary
- Layout file preparation (reference the layout template)
- Separation and detection parameters
- Troubleshooting (low signal, high background, peak resolution)
- Reference the official Jess protocol PDF

**Section 4: SOPs Index**
For each SOP in the folder, provide:
- Brief purpose (1-2 sentences)
- Key parameters (volumes, temperatures, incubation times)
- Critical steps that commonly go wrong
SOPs available: {', '.join(f for f in file_listing if 'SOP' in f or 'Sample preparation' in f)}

**Section 5: Cross-Method Utilities**
- Sample homogenization (different tissue types)
- Cell lysate preparation best practices  
- Protein quantification decision tree
- Common calculation formulas (dilution factors, concentration from standard curve)

**Section 6: System Prompt Addition**
Write a paragraph that should be appended to any method-specific system prompt when 
general protocols knowledge is needed. This defines the assistant's cross-method 
expertise.

=== FILES IN GENERAL PROTOCOLS FOLDER ===
{file_list_str}

=== READABLE FILE CONTENTS ===
{db_context}

=== FORMAT GUIDELINES ===
- Use clear markdown headers
- Include practical bench-ready details
- Reference specific files from the database where relevant
- Include warning markers (⚠️) for critical steps
- Be concise but complete — this loads into every session context
"""

    print("\nGenerating general_methods_assistant content...")
    content = generate_with_retry(prompt, "general_methods_assistant")
    print(f"Generated {len(content)} chars")

    # 4. Save content locally first
    local_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'general_methods_assistant.md')
    with open(local_path, 'w') as f:
        f.write(content)
    print(f"Saved locally to {local_path}")

    # Try to find or create doc in general protocols folder
    q = f"'{gp_id}' in parents and name='general_methods_assistant' and trashed=false"
    existing = svc.files().list(q=q, fields='files(id,mimeType)').execute().get('files', [])

    if existing and 'document' in existing[0]['mimeType']:
        doc_id = existing[0]['id']
        print(f"Found existing doc: {doc_id}, writing...")
        # Truncate if too long
        if len(content) > 30000:
            content = content[:30000] + "\n\n[... truncated for context window efficiency ...]"
            print(f"Truncated to {len(content)} chars")
        asyncio.run(append_doc_text(doc_id, content))
        print(f"✓ Written to Google Doc {doc_id}")
    else:
        print("No Google Doc found in 'general protocols' folder.")
        print("Please create an empty Google Doc named 'general_methods_assistant' in the")
        print("'general protocols' folder (shared with the service account), then re-run.")
        print(f"\nContent saved to: {local_path}")
        print("Or run: python scripts/fill_general_methods.py --write-to DOC_ID")

    print(f"✓ general_methods_assistant written! ({len(content)} chars)")
    print(f"  Doc ID: {doc_id}")


if __name__ == '__main__':
    main()
