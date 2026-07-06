import streamlit as st
from utils.file_handler import save_uploaded_file
from utils.pdf_loader import extract_pdf
from utils.chunking import createChunks
from utils.embeddings import getEmbeddings
from utils.vector_store import upload_to_pinecone
from utils.retrieval import get_retriever
from utils.prompt import get_rag_prompt
import math
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from config import GOOGLE_API_KEY

st.set_page_config(page_title="DocMind AI")
st.title("📚 DocMind AI")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)
if uploaded_file:
    file_path = save_uploaded_file(uploaded_file)
    st.success("File Uploaded successfully")
    result = extract_pdf(file_path)
    st.write("### Total Pages")
    st.write(result["pages"])
    chunks = createChunks(result["text"])

# estimated time based upon chunks
    total_batches = math.ceil(len(chunks) / 5)
    estimated_seconds = (total_batches - 1) * 62
    estimated_minutes = estimated_seconds // 60
    st.info(f"⏳ Estimated time to index: ~{estimated_minutes} minutes ({total_batches} batches)")

# calling upload_to_pinecone() => hitting the API quota again and again, so fixing that bug 
    if "uploaded_file_name" not in st.session_state or st.session_state.uploaded_file_name != uploaded_file.name:
        with st.spinner("Embedding and uploading to Pinecone... (this may take a while for large documents)"):
            embedding_model = getEmbeddings()
            upload_to_pinecone(chunks, embedding_model)
        
# Save the file name and embedding model to session_state so we don't redo this
        st.session_state.uploaded_file_name = uploaded_file.name
        st.session_state.embedding_model = embedding_model

    else:
        embedding_model = st.session_state.embedding_model

    index = get_retriever()
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY
    )

# fetch context from Pinecone
    def retrieve_context(question: str) -> str:
        query_vector = embedding_model.embed_query(question)
        response = index.query(
            vector=query_vector,
            top_k=3,
            include_metadata=True
        )
        return "\n\n".join([match["metadata"]["text"] for match in response["matches"]])

    prompt = get_rag_prompt()

# Build the sequential and parallel chain using LangChain Runnables (LCEL)
    rag_chain = (
        {
            "context": retrieve_context,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    query = st.text_input("Ask a question about the document:")
    if query:
        with st.spinner("Generating answer..."):
            answer = rag_chain.invoke(query)
        st.write("### Answer:")
        st.info(answer)
