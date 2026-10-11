import os
import hashlib
from pathlib import Path
UPLOAD_DIR = "uploads" 

os.makedirs(UPLOAD_DIR, exist_ok=True)

# temp upload -> ./uploads
def save_uploaded_file(uploaded_file):
    file_bytes = uploaded_file.getbuffer()
    content_hash = hashlib.sha256(bytes(file_bytes)).hexdigest()
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(file_bytes)
# fetches data from RAM buffer memory 
    return file_path, content_hash