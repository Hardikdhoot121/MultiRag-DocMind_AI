import os
from pathlib import Path
UPLOAD_DIR = "uploads" 
# temp upload -> ./uploads
def save_uploaded_file(uploaded_file):
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
# fetches data from RAM buffer memory 
    return file_path