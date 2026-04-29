#!/usr/bin/env python3
"""Read HCR FISH database files for method_support generation."""
import sys, os, io
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httplib2
httplib2.CA_CERTS = '/etc/ssl/certs/ca-certificates.crt'

from src.google_client import _get_service
from src.config import DRIVE_ROOT_FOLDER_ID
import docx

svc = _get_service('drive', 'v3')

def download_docx_content(file_id):
    content = svc.files().get_media(fileId=file_id).execute()
    doc = docx.Document(io.BytesIO(content))
    text = '\n'.join([p.text for p in doc.paragraphs if p.text.strip()])
    return text

def find_folder(parent_id, name):
    q = f"'{parent_id}' in parents and name='{name}' and trashed=false and mimeType='application/vnd.google-apps.folder'"
    res = svc.files().list(q=q, fields='files(id)').execute().get('files', [])
    return res[0]['id'] if res else None

# Navigate to HCR FISH > database > HCR FISH protocols
hcr_id = find_folder(DRIVE_ROOT_FOLDER_ID, 'HCR FISH')
db_id = find_folder(hcr_id, 'database')
proto_id = find_folder(db_id, 'HCR FISH protocols')
v3_id = find_folder(proto_id, 'V.3 probes')

# Read Cells fixation protocol.docx
q = f"'{proto_id}' in parents and name contains '.docx' and trashed=false"
docxs = svc.files().list(q=q, fields='files(id,name)').execute().get('files', [])
for d in docxs:
    print(f'--- {d["name"]} ---')
    text = download_docx_content(d['id'])
    print(text[:2000])
    print(f'[total: {len(text)} chars]\n')

# Read V.3 probes .docx files
q = f"'{v3_id}' in parents and name contains '.docx' and trashed=false"
docxs2 = svc.files().list(q=q, fields='files(id,name)', orderBy='name').execute().get('files', [])
all_protocols = []
for d in docxs2:
    text = download_docx_content(d['id'])
    lines = text.split('\n')
    all_protocols.append({'name': d['name'], 'text': text, 'lines': len(lines), 'chars': len(text)})
    print(f'--- {d["name"]} ---')
    print(f'  Lines: {len(lines)}, Chars: {len(text)}')
    # Show first 500 chars
    print(text[:500])
    print('...\n')

# Also read calculation sheet names for summary
calc_id = find_folder(db_id, 'HCR FISH calculation sheets')
q = f"'{calc_id}' in parents and trashed=false"
calcs = svc.files().list(q=q, fields='files(id,name)', orderBy='name').execute().get('files', [])
print('\n=== Calculation Sheets Available ===')
for c in calcs:
    if not c['name'].startswith('~$'):
        print(f'  - {c["name"]}')
