import os 
from concurrent.futures import ThreadPoolExecutor  # parallel threads for execution , reading task not generation of embeddings 
from utils.image_loader import describe_image
from utils.pdf_loader import extract_pdf
from utils.error_handler import handle_errors

def _process_single_file(file_path):
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    if ext == ".pdf":
        return extract_pdf(file_path)
    elif ext in [".jpg", ".jpeg", ".png"]:
        return describe_image(file_path)
    else:
        print(f"unsupported file type: {ext}")
        return []

@handle_errors(fallback=[])
def file_router(file_paths):
    all_documents = []
    # Parallelize file reading using ThreadPoolExecutor (max 2 workers)
    with ThreadPoolExecutor(max_workers=min(2, len(file_paths) or 1)) as executor:
        results = executor.map(_process_single_file, file_paths)
        for docs in results:
            if docs:
                all_documents.extend(docs)
    return all_documents