# MultiRAG

Python • Streamlit • LangChain • Gemini • Pinecone • PyMuPDF • Multi-Modal RAG

**[🌐 View Live Streamlit App Here](https://multirag-docmind-ai.streamlit.app)**

MultiRAG is an enterprise-grade, multi-modal Retrieval-Augmented Generation (RAG) pipeline. It allows users to upload multiple PDFs and images simultaneously, semantically search through their contents, and generate accurate, source-aware answers using Google's Gemini models and Pinecone's vector database.

---

## Table of Contents

- [Features](#features)
- [Technology Stack](#technology-stack)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
- [Installation](#installation)
- [Environment Variables](#environment-variables)
- [Folder Structure](#folder-structure)
- [Performance Optimizations](#performance-optimizations)
- [Current Features](#current-features)
- [Upcoming Features](#upcoming-features)
- [External API Integrations](#external-api-integrations)
- [Author](#author)
- [License](#license)

---

## Features

- **Multi PDF Upload**: Process and index multiple PDF documents in a single session.
- **Multi Image Upload**: Native support for processing `.png`, `.jpg`, and `.jpeg` files.
- **Multi-Modal Retrieval-Augmented Generation**: Unified querying across text documents and image descriptions.
- **Semantic Search**: Highly accurate vector-based search using Pinecone.
- **Google Gemini Models**: Utilizing the latest Gemini Flash and Vision models for generation and understanding.
- **Gemini Embeddings**: High-dimensional text embeddings for optimal retrieval accuracy.
- **PyMuPDF PDF Parsing**: Extremely fast and reliable PDF text extraction.
- **Image Understanding**: Converting images to textual descriptions using Gemini Vision.
- **Recursive Character Text Splitting**: Intelligent chunking of large texts without breaking semantic context.
- **Metadata Preservation**: Automatic tracking of source files, document types, and page numbers.
- **LangChain LCEL Pipeline**: Streamlined, declarative pipeline for the retrieval and generation chain.
- **Source-aware Responses**: The LLM strictly answers based on retrieved context.
- **Duplicate File Caching**: Intelligent session-state hashing to prevent redundant embedding of unchanged files.
- **Rate Limit Handling**: Built-in batching and sleep intervals to respect API free-tier quotas.
- **Streamlit Interface**: Clean, interactive frontend for seamless user experience.

---

## Technology Stack

| Component | Technology |
|---|---|
| **Language** | Python 3.12+ |
| **Frontend** | Streamlit |
| **Framework** | LangChain |
| **Generation LLM** | Gemini Flash Latest |
| **Embedding Model** | Google Gemini Embedding 001 |
| **Image Understanding** | Gemini 3.1 Flash Lite (Vision) |
| **Vector Database** | Pinecone |
| **PDF Parser** | PyMuPDF |
| **Chunking** | RecursiveCharacterTextSplitter |

---

## Architecture

```text
               Upload
          PDF  PDF  IMG  IMG
                    │
                    ▼
               File Router
          ┌─────────┴─────────┐
          │                   │
      PDF Loader        Image Loader
          │                   │
     PyMuPDF           Gemini Vision
          │                   │
      Documents       Image Description
          └─────────┬─────────┘
                    ▼
            Merge Documents
                    ▼
               Chunking
                    ▼
             Embedding Model
                    ▼
               Pinecone

────────────────────────────────────────

              User Question
                    ▼
             Query Embedding
                    ▼
         Pinecone Similarity Search
                    ▼
        Top-K Relevant Chunks
                    ▼
          ChatPromptTemplate
                    ▼
          Gemini Flash Latest
                    ▼
             Final Answer
```

---

## Getting Started

### Prerequisites

Before you begin, ensure you have the following installed and configured:

- **Python 3.12+**
- **Git**
- **Google Gemini API Key**: [Get it here](https://aistudio.google.com/app/apikey)
- **Pinecone API Key**: [Get it here](https://www.pinecone.io/)

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Hardikdhoot121/MultiRag-DocMind_AI.git
cd MultiRag_pipeline
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
- **Windows**:
  ```bash
  .\venv\Scripts\activate
  ```
- **macOS/Linux**:
  ```bash
  source venv/bin/activate
  ```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the root directory and add the following keys.

| Variable | Description |
|---|---|
| `GOOGLE_API_KEY` | Your Google Gemini API key for embeddings, vision, and text generation. |
| `PINECONE_API_KEY` | Your Pinecone database API key for vector storage. |
| `PINECONE_INDEX_NAME` | The exact name of the Pinecone index you created in your dashboard. |

To start the application, run:
```bash
streamlit run app.py
```

---

## Folder Structure

```text
MultiRAG/
├── models/
│   └── vector_store.py     # Pinecone index management and batch uploading
├── routes/
│   └── file_router.py      # Dispatches files to appropriate loaders based on extension
├── utils/
│   ├── file_handler.py     # Saves Streamlit UploadedFile objects to temporary disk
│   ├── chunking.py         # Handles RecursiveCharacterTextSplitter logic
│   ├── embeddings.py       # Initializes Gemini embeddings
│   ├── image_loader.py     # Uses Gemini Vision to convert images to Document objects
│   ├── pdf_loader.py       # Uses PyMuPDF to extract text into Document objects
│   ├── prompt.py           # Defines the RAG ChatPromptTemplate
│   └── retrieval.py        # Initializes the Pinecone retriever
├── uploads/                # Temporary directory for file processing
├── app.py                  # Main Streamlit application entry point
├── config.py               # Environment variable loader
├── requirements.txt        # Project dependencies
├── .env                    # Secret keys
└── README.md               # Project documentation
```

---

## Performance Optimizations

- **Duplicate File Detection**: The application calculates a stable hash of the uploaded filenames. If the hash remains unchanged between queries, the system completely bypasses the expensive embedding and upload phases, utilizing the cached Pinecone index instead.
- **Rate Limit Batching**: Free-tier Google API rate limits are strictly managed using batch processing and strategic `time.sleep()` intervals, ensuring the pipeline does not fail with `429 RESOURCE_EXHAUSTED` errors.
- **Dynamic Routing**: Only the necessary loaders are invoked based on the file type, preventing unnecessary processing overhead.

---

## Current Features

- Multi-modal upload support (Up to 4 files: PDFs & Images).
- Automated document routing and unified formatting.
- Image-to-text semantic translation via Gemini Vision.
- Stateful Streamlit UI with intelligent caching.
- Pinecone vector database integration with automatic stale-data cleanup.
- LCEL-based Q&A generation pipeline.

---

## Upcoming Features

- **Chat History**: Maintaining conversation memory for follow-up questions.
- **Expanded File Support**: Adding support for `.docx`, `.txt`, and `.csv` files.
- **Cloud Storage Integration**: Direct ingestion from Google Drive and AWS S3.

---

## External API Integrations

MultiRAG relies on a robust backend architecture that orchestrates multiple external APIs:
- **Google Gemini API**: Makes `POST` requests to generate high-dimensional text embeddings, process images using Gemini Vision, and generate the final RAG text responses.
- **Pinecone API**: Makes batched `POST` requests (upserts) to securely store vector data and handles `POST` query requests to perform sub-second cosine similarity searches across the document database.

---

## Author

**Hardik Dhoot**
- **GitHub**: [Hardikdhoot121](https://github.com/Hardikdhoot121)

---

## License

Distributed under the MIT License. See `LICENSE` for more information.
