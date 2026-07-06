from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import GOOGLE_API_KEY

def getEmbeddings():
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    return embedding_model