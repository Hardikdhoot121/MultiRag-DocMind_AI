from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP
def createChunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    chunks = []
    docs_to_split = []

    for doc in documents:
        # Table data ko break hone se bachao taaki Markdown table rows intact rahein
        if doc.metadata.get("content_type") == "table" and len(doc.page_content) <= (CHUNK_SIZE * 2):
            chunks.append(doc)
        else:
            docs_to_split.append(doc)

    if docs_to_split:
        chunks.extend(splitter.split_documents(docs_to_split))

    return chunks