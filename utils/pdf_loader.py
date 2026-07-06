import fitz
def extract_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    complete_text = ""
    for page in doc:
        complete_text += page.get_text()
    return {
        "text": complete_text,
        "pages": len(doc)
}
#fitz function internally reads the document , like OS open file , then read byte , parse pdf , create doc object and return 
