import fitz
import os 
from langchain_core.documents import Document
def extract_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    documents=[]
    complete_text = ""
    for page_num, page in enumerate(doc):
        text=page.get_text()
        if(text.strip()):
            documents.append(Document(
                page_content=text,
                metadata={
                    "page_number": page_num + 1,
                    "source": os.path.basename(pdf_path),
                    "doc_type": "pdf"
                }
            ))
    return documents
#fitz function internally reads the document , like OS open file , then read byte , parse pdf , create doc object and return 
