import base64
import os
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from config import GOOGLE_API_KEY
from utils.error_handler import handle_errors

@handle_errors(fallback=[])
def describe_image(image_path):
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
        
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    model = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite", 
        google_api_key=GOOGLE_API_KEY,
        max_retries=4
    )
    message = HumanMessage(
        content=[
            {
                "type": "text", 
                "text": (
                    "Analyze this image and extract content strictly into these 3 sections:\n\n"
                    "### [TEXT_CONTENT]\nExtract all readable text verbatim.\n\n"
                    "### [TABLE_DATA]\nConvert all tabular data, receipts, or numerical rows into Markdown tables (| col | col |). If none, write 'None'.\n\n"
                    "### [VISUAL_DIAGRAM]\nExplain any diagrams, flowcharts, or architecture in 'Entity A -> Entity B' relationship format. If none, write 'None'."
                )
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
            },
        ]
    )
    try:
        response = model.invoke([message])
        if isinstance(response.content, str):
            content_text = response.content
        else:
            content_text = ""
            for part in response.content:
                if isinstance(part, dict) and "text" in part:
                    content_text += part["text"]
    except Exception as e:
        content_text = f"[Image description temporary error: {e}]"

    # Parse into semantic documents
    markers = [
        ("[TEXT_CONTENT]", "text"),
        ("[TABLE_DATA]", "table"),
        ("[VISUAL_DIAGRAM]", "diagram")
    ]
    documents = []
    current_type = "general"
    current_lines = []

    for line in content_text.splitlines():
        matched = False
        for marker, ctype in markers:
            if marker in line:
                if current_lines:
                    text_block = "\n".join(current_lines).strip()
                    if text_block and text_block.lower() != "none":
                        documents.append(Document(
                            page_content=text_block,
                            metadata={"source": os.path.basename(image_path), "doc_type": "image", "content_type": current_type, "image_path": image_path}
                        ))
                current_type = ctype
                current_lines = []
                matched = True
                break
        if not matched:
            current_lines.append(line)

    if current_lines:
        text_block = "\n".join(current_lines).strip()
        if text_block and text_block.lower() != "none":
            documents.append(Document(
                page_content=text_block,
                metadata={"source": os.path.basename(image_path), "doc_type": "image", "content_type": current_type, "image_path": image_path}
            ))

    if not documents:
        documents = [Document(
            page_content=content_text,
            metadata={"source": os.path.basename(image_path), "doc_type": "image", "content_type": "general", "image_path": image_path}
        )]

    return documents
