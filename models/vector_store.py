from pinecone import Pinecone
from config import PINECONE_INDEX_NAME, PINECONE_API_KEY
import time 

def upload_to_pinecone(chunks, embedding_model):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    
    index.delete(delete_all=True)

    texts = [doc.page_content for doc in chunks]
    embeddings = []
    for i in range (0,len(texts),5):
        embeddings+=embedding_model.embed_documents(texts[i:i+5])
        if(i+5<len(texts)):
            time.sleep(62)
    
    vectors = []
    for i, (text, emb, doc) in enumerate(zip(texts, embeddings, chunks)):
        vectors.append({
            "id": f"chunk_{i}_{hash(text)}",
            "values": emb,
            "metadata": {
                "text": text,
                **doc.metadata
            }
        })
    
    # Upsert in batches of 100
    batch_size = 10
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)
        
    return index