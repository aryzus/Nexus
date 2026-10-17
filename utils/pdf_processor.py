import pymupdf
import os
from pathlib import Path

def extract_text_from_pdf(pdf_path):
    doc = pymupdf.open(pdf_path)
    text_by_page = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        if text.strip():
            text_by_page.append({
                'page': page_num + 1,
                'text': text.strip(),
                'source': Path(pdf_path).name
            })
    
    doc.close()
    print(f"Extracted {len(text_by_page)} pages from {Path(pdf_path).name}")
    return text_by_page

def prepare_documents_for_rag(uploaded_files, temp_dir="temp_pdfs"):
    os.makedirs(temp_dir, exist_ok=True)
    all_pages = []
    doc_summaries = {}
    
    for uploaded_file in uploaded_files:
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        pages = extract_text_from_pdf(temp_path)
        all_pages.extend(pages)
        
        doc_summaries[uploaded_file.name] = {
            'pages': len(pages),
            'size_mb': round(uploaded_file.size / 1024 / 1024, 2),
            'preview': pages[0]['text'][:300] if pages else ""
        }
    
    return all_pages, doc_summaries

def pages_to_chunks(pages, chunk_size=800, overlap=100):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " "]
    )
    
    all_chunks = []
    all_metadata = []
    
    for page in pages:
        chunks = splitter.split_text(page['text'])
        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadata.append({
                'source': page['source'],
                'page': page['page']
            })
    
    print(f"Created {len(all_chunks)} chunks from {len(pages)} pages")
    return all_chunks, all_metadata