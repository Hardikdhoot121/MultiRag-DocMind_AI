import os 
from utils.image_loader import describe_image
from utils.pdf_loader import extract_pdf
def file_router(file_paths):
    all_documents=[]
    for file_path in file_paths:
        _,ext=os.path.splitext(file_path)
        ext=ext.lower()

        if ext==".pdf":
            docs=extract_pdf(file_path)

        elif ext in [".jpg", ".jpeg", ".png"]:
            docs=describe_image(file_path)
        else:
            print(f"unsupported file type:{ext}") 

        all_documents.extend(docs)
    return all_documents

        