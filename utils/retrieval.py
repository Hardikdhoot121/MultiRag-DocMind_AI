from pinecone import Pinecone
from config import PINECONE_INDEX_NAME, PINECONE_API_KEY

def get_retriever():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(PINECONE_INDEX_NAME)
