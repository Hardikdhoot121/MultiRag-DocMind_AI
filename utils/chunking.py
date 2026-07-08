from langchain_text_splitters import RecursiveCharacterTextSplitter
def createChunks(text):
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=500,
        separators=["\n\n", "\n", ".", " ",""]
    )
    chunks=splitter.split_documents(text)
    return chunks 