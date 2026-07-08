from langchain_core.prompts import ChatPromptTemplate
def get_rag_prompt():
    prompt = ChatPromptTemplate([
        ("system", 
            "You are a helpful assistant. Answer the user's question using ONLY the provided context below.\n"
            "If the answer is not present in the context, respond with exactly: "
            "\"I couldn't find that information in the uploaded documents.\"\n"
            "Keep answers concise and clear. Do not hallucinate or make up details.\n\n"
            "Context:\n{context}"
        ),
        ("human", "{question}")
    ])
    return prompt
