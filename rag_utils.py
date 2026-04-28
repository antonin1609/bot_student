from pypdf import PdfReader
import faiss
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def get_embedder():
    return SentenceTransformer(EMBED_MODEL)

def extract_pdf_pages(uploaded_files):
    pages = []
    for file in uploaded_files:
        reader = PdfReader(file)
        for page_num, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append({"text": text, "source": file.name, "page": page_num})
    return pages

def chunk_text(text, chunk_size=1200, overlap=200):
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break
        start = max(0, end - overlap)
    return chunks

def build_documents(uploaded_files, progress_callback=None):
    pages = extract_pdf_pages(uploaded_files)
    docs = []
    total = max(1, len(pages))
    for i, item in enumerate(pages, start=1):
        if progress_callback:
            progress_callback(i / total)
        parts = chunk_text(item["text"])
        for j, part in enumerate(parts):
            docs.append({
                "text": part,
                "source": item["source"],
                "page": item["page"],
                "chunk_id": j
            })
    if progress_callback:
        progress_callback(1.0)
    return docs

def create_faiss_index(docs, embedder=None, progress_callback=None):
    if embedder is None:
        embedder = get_embedder()
    texts = [d["text"] for d in docs]
    total = max(1, len(texts))
    embeddings_list = []
    batch_size = 32
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        emb = embedder.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
        embeddings_list.append(emb)
        if progress_callback:
            progress_callback(min(1.0, (start + len(batch)) / total))
    embeddings = __import__("numpy").vstack(embeddings_list).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    if progress_callback:
        progress_callback(1.0)
    return index

def retrieve(question, docs, index, embedder=None, top_k=5, source_filter=None):
    if embedder is None:
        embedder = get_embedder()
    if source_filter and source_filter != "Tous les documents":
        filtered = [(i, d) for i, d in enumerate(docs) if d["source"] == source_filter]
        if not filtered:
            return []
        filtered_ids = [i for i, _ in filtered]
        filtered_docs = [d for _, d in filtered]
        texts = [d["text"] for d in filtered_docs]
        temp_index = faiss.IndexFlatIP(index.d)
        emb = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        temp_index.add(emb)
        q_emb = embedder.encode([question], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        _, ids = temp_index.search(q_emb, top_k)
        results = []
        for idx in ids[0]:
            if idx != -1:
                results.append(filtered_docs[idx])
        return results

    q_emb = embedder.encode([question], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    _, ids = index.search(q_emb, top_k)
    results = []
    for idx in ids[0]:
        if idx != -1:
            results.append(docs[idx])
    return results

def format_context(results):
    blocks = []
    for n, r in enumerate(results, start=1):
        blocks.append(f"[Extrait {n}] Source: {r['source']} | Page: {r['page']}\n{r['text']}")
    return "\n\n".join(blocks)

def format_sources(results):
    seen = []
    for r in results:
        label = f"{r['source']} (page {r['page']})"
        if label not in seen:
            seen.append(label)
    return seen