import base64
import os
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from config import GOOGLE_API_KEY

def describe_image(image_path):
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
        
    # converts image byte to a base64 encoded string
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    model = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite", 
        google_api_key=GOOGLE_API_KEY
    )
    message = HumanMessage(
        content=[
            {
                "type": "text", 
                "text": "Describe this image in full detail. Extract all visible text, diagrams, tables, flowcharts, and core concepts."
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
            },
        ]
    )
    response = model.invoke([message])

    if isinstance(response.content, str):
        content_text = response.content
    else:
        content_text = ""
        for part in response.content:
            if isinstance(part, dict) and "text" in part:
                content_text += part["text"]

    # wrap it with meta data and return as list format
    doc = Document(
        page_content=content_text,
        metadata={
            "source": os.path.basename(image_path),
            "doc_type": "image"
        }
    )
     
    return [doc]
