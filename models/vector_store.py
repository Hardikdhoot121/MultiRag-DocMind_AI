from pinecone import Pinecone
from config import PINECONE_INDEX_NAME, PINECONE_API_KEY, EMBED_BATCH_SIZE, UPSERT_BATCH_SIZE
from utils.error_handler import handle_errors
import hashlib
import time

def _embed_with_retry(texts, embedding_model, batch_size=EMBED_BATCH_SIZE, max_retries=4):
    """Embed texts in batches with exponential backoff retry."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        for attempt in range(max_retries):
            try:
                embs = embedding_model.embed_documents(batch)
                all_embeddings.extend(embs)
                break
            except Exception as e:
                wait_time = (2 ** attempt) * 5  # 5s, 10s, 20s, 40s
                print(f"[RETRY {attempt+1}/{max_retries}] Embedding failed: {e}. Waiting {wait_time}s...")
                if attempt == max_retries - 1:
                    print(f"[ERROR] Embedding batch {i} permanently failed after {max_retries} retries.")
                    all_embeddings.extend([None] * len(batch))
                else:
                    time.sleep(wait_time)
    return [e for e in all_embeddings if e is not None]

def _upsert_with_retry(index, batch, namespace, max_retries=3):
    """Upsert vectors with retry logic."""
    for attempt in range(max_retries):
        try:
            index.upsert(vectors=batch, namespace=namespace)
            return
        except Exception as e:
            wait_time = (2 ** attempt) * 2
            print(f"[RETRY {attempt+1}/{max_retries}] Upsert failed: {e}. Waiting {wait_time}s...")
            if attempt < max_retries - 1:
                time.sleep(wait_time)

@handle_errors(reraise=True)
def upload_to_pinecone(chunks, embedding_model, namespace="docmind-default", delete_old=True):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    
    if delete_old:
        try:
            index.delete(delete_all=True, namespace=namespace)
        except Exception:
            pass

    texts = [doc.page_content for doc in chunks]
    embeddings = _embed_with_retry(texts, embedding_model)
    
    if len(embeddings) != len(texts):
        print(f"[ERROR] Embedding count mismatch: {len(embeddings)} vs {len(texts)}")
        return index

    vectors = []
    for i, (text, emb, doc) in enumerate(zip(texts, embeddings, chunks)):
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        vectors.append({
            "id": f"chunk_{content_hash}",
            "values": emb,
            "metadata": {
                "text": text,
                **doc.metadata
            }
        })
    
    # Upsert in batches of 50
    batch_size = UPSERT_BATCH_SIZE
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        _upsert_with_retry(index, batch, namespace)
        
    return index