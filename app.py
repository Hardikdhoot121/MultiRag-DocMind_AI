import streamlit as st
from utils.file_router import file_router
from utils.file_handler import save_uploaded_file
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

uploaded_files = st.file_uploader(
    "Upload PDF or Image (MAX 4)",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files:
    if len(uploaded_files) > 4:
        st.error("You can only upload a maximum of 4 files")
    else :
        file_paths=[] #This will be send to router 
        for uploaded_file in uploaded_files:
            temp_path=save_uploaded_file(uploaded_file)
            file_paths.append(temp_path)

        st.success("All Files Uploaded successfully")
        
        # check -> if previously uploaded then just fetch the embeddings 
        #       -> else run the entire pipeline 
        current_files_hash = ",".join(sorted([f.name for f in uploaded_files]))
        if "uploaded_files_hash" not in st.session_state or st.session_state.uploaded_files_hash != current_files_hash:
     
            result = file_router(file_paths)
            st.write(f"### Total Documents: {len(result)}")
            chunks = createChunks(result)

            total_batches = math.ceil(len(chunks) / 5)
            estimated_seconds = (total_batches - 1) * 15
            estimated_minutes = estimated_seconds // 60
            st.info(f"⏳ Estimated time to index: ~{estimated_minutes} minutes ({total_batches} batches)")

            with st.spinner("Embedding and uploading to Pinecone..."):
                embedding_model = getEmbeddings()
                upload_to_pinecone(chunks, embedding_model)

            st.session_state.uploaded_files_hash = current_files_hash
            st.session_state.embedding_model = embedding_model
            st.success("Documents indexed successfully!")
        else:
            embedding_model = st.session_state.embedding_model

        # RETRIEVAL (runs every timedef , but is instant)
        index = get_retriever()
        llm = ChatGoogleGenerativeAI(
            model="gemini-flash-latest",
            google_api_key=GOOGLE_API_KEY
        )

        def retrieve_context(question: str) -> str:
            query_vector = embedding_model.embed_query(question)
            response = index.query(
                vector=query_vector,
                top_k=3,
                include_metadata=True
            )
            return "\n\n".join([match["metadata"]["text"] for match in response["matches"]])
        # AUGEMENTATION 
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
        # GENERATION
        query = st.text_input("Ask a question about the documents:")
        if query:
            with st.spinner("Generating answer..."):
                answer = rag_chain.invoke(query)
            st.write("### Answer:")
            st.info(answer)


