import streamlit as st
import hashlib
from routes.file_router import file_router
from utils.file_handler import save_uploaded_file
from utils.chunking import createChunks
from utils.embeddings import getEmbeddings
from models.vector_store import upload_to_pinecone
from utils.retrieval import get_retriever
from utils.prompt import get_rag_prompt
import math
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from config import GOOGLE_API_KEY, TOP_K, SCORE_THRESHOLD

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
        file_paths = []
        file_hashes = []
        for uploaded_file in uploaded_files:
            temp_path, content_hash = save_uploaded_file(uploaded_file)
            file_paths.append(temp_path)
            file_hashes.append(content_hash)

        st.success("All Files Uploaded successfully")
        
        combined = "|".join(sorted(file_hashes))
        current_files_hash = hashlib.sha256(combined.encode()).hexdigest()
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
                upload_to_pinecone(chunks, embedding_model, namespace=current_files_hash)

            st.session_state.uploaded_files_hash = current_files_hash
            st.session_state.embedding_model = embedding_model
            st.success("Documents indexed successfully!")
        else:
            embedding_model = st.session_state.embedding_model

        index = get_retriever()
        llm = ChatGoogleGenerativeAI(
            model="gemini-flash-latest",
            google_api_key=GOOGLE_API_KEY,
            max_retries=4
        )

        uploaded_doc_names = [f.name for f in uploaded_files]
        selected_doc = st.selectbox(
            "🎯 Filter by Document (Optional):",
            options=["All Documents"] + uploaded_doc_names,
            index=0
        )
        def retrieve_and_format_context(inputs: dict) -> dict:
            question = inputs["question"]
            doc_filter = inputs.get("doc_filter", "All Documents")
            
            # Metadata filters with metadata 
            pinecone_filter = None
            if doc_filter != "All Documents":
                pinecone_filter = {"source": {"$eq": doc_filter}}

            query_vector = embedding_model.embed_query(question)
            response = index.query(
                vector=query_vector,
                top_k=TOP_K,
                include_metadata=True,
                namespace=st.session_state.uploaded_files_hash,
                filter=pinecone_filter
            )
            
            filtered_matches = [
                m for m in response.get("matches", []) 
                if m.get("score", 0) >= SCORE_THRESHOLD
            ]
            
            if not filtered_matches:
                return {"context": "", "sources": []}
            
            context_blocks = []
            sources = []
            for i, match in enumerate(filtered_matches, 1):
                meta = match["metadata"]
                source_file = meta.get("source", "Uploaded Document")
                doc_type = meta.get("doc_type", "pdf")
                content_type = meta.get("content_type", "text")
                
                if doc_type == "image":
                    tag = f"[Source {i} (Image: {source_file} | Type: {content_type.upper()})]"
                    citation = f"🖼️ **{source_file}** ({content_type.upper()}) (Relevance: {match.get('score', 0):.2f})"
                else:
                    page_num = meta.get("page_number", meta.get("page", None))
                    page_info = f", Page {page_num}" if page_num is not None else ""
                    tag = f"[Source {i} (Document: {source_file}{page_info})]"
                    citation = f"📄 **{source_file}**{page_info} (Relevance: {match.get('score', 0):.2f})"

                context_blocks.append(f"{tag}\n{meta.get('text', '')}")
                sources.append(citation)
                
            return {
                "context": "\n\n".join(context_blocks),
                "sources": sources
            }

        prompt = get_rag_prompt()
        output_parser = StrOutputParser()

        # clean LCEL Chain Pipeline
        def run_rag_pipeline(question: str, doc_filter: str):
            retrieved = retrieve_and_format_context({"question": question, "doc_filter": doc_filter})
            if not retrieved["context"]:
                return "⚠️ No relevant information found in the uploaded documents for this query.", []
            
            chain = prompt | llm | output_parser
            answer = chain.invoke({
                "context": retrieved["context"],
                "question": question
            })
            return answer, retrieved["sources"]

        query = st.text_input("Ask a question about the documents:")
        if query:
            with st.spinner("Searching documents & generating answer..."):
                try:
                    answer, sources = run_rag_pipeline(query, selected_doc)
                    st.write("### Answer:")
                    if answer.startswith("⚠️"):
                        st.warning(answer)
                    else:
                        st.info(answer)
                        if sources:
                            with st.expander("📍 View Retrieved Sources & Citation Scores"):
                                for src in sources:
                                    st.write(src)
                except Exception as e:
                    st.error("Google Gemini API is currently experiencing high demand (503 Service Unavailable). Please wait a few seconds and try again.")
                    st.caption(f"Error details: {e}")


