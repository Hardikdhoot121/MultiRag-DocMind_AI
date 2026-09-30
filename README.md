# MultiRAG

Python • Streamlit • LangChain • Gemini • Pinecone • PyMuPDF • Multi-Modal RAG

**[🌐 View Live Streamlit App Here](https://your-app-url-goes-here.streamlit.app/)**

MultiRAG is a multi-modal Retrieval-Augmented Generation (RAG) pipeline built to bridge structured document extraction and visual content analysis. It enables users to upload multiple PDFs and images (`.png`, `.jpg`, `.jpeg`), indexes them into a unified vector space via Pinecone, and answers context-specific questions with source-bounded responses using Google's Gemini models.

---

## Table of Contents

- [Features](#features)
- [Technology Stack](#technology-stack)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
- [Installation](#installation)
- [Environment Variables](#environment-variables)
- [Folder Structure](#folder-structure)
- [Performance & Design Trade-offs](#performance--design-trade-offs)
- [Current Implementation Status](#current-implementation-status)
- [Future Roadmap & Improvements](#future-roadmap--improvements)
- [External API Integrations](#external-api-integrations)
- [Author](#author)
- [License](#license)

---

## Features

- **Multi-File Batch Processing**: Ingest and index multiple PDF documents and images within a single session.
- **Image-to-Text Semantic Translation**: Transforms visual inputs (`.png`, `.jpg`, `.jpeg`) into dense text descriptions using `gemini-3.1-flash-lite` before indexing.
- **Unified Vector Space**: Combines PyMuPDF document chunks and Gemini Vision image descriptions into a single retriever pipeline.
- **Sub-Second Similarity Search**: Vector retrieval powered by Pinecone's cosine distance index.
- **LangChain LCEL Chain**: Modern, pipe-based chain (`retriever | prompt | llm | parser`) for clear control flow and execution tracking.
- **Metadata Tagging**: Preserves source file names and document structures across chunks for context assembly.
- **Rate-Limit Resilient**: Embedded exponential retry logic (`max_retries=4`) and batch sleep intervals to prevent Gemini API quota throttling.
- **Session-Based UI Caching**: Prevents redundant vector upserts during active Streamlit user sessions when document selections remain identical.

---

## Technology Stack

| Component | Technology | Role / Model |
|---|---|---|
| **Language** | Python 3.12+ | Core runtime |
| **Frontend UI** | Streamlit | Web interface & session state management |
| **Orchestration** | LangChain (LCEL) | Retrieval & generation chain wiring |
| **Generation LLM** | Google Gemini (`gemini-flash-latest`) | Context-bounded response generation |
| **Embedding Model** | Google Generative AI Embeddings (`text-embedding-004`) | 768-dimensional text vector generation |
| **Vision Model** | Google Gemini (`gemini-3.1-flash-lite`) | Image-to-text description extraction |
| **Vector Database** | Pinecone DB | Vector indexing and top-K similarity search |
| **PDF Parser** | PyMuPDF (`fitz`) | Fast text extraction from PDF pages |
| **Text Splitter** | `RecursiveCharacterTextSplitter` | Chunking (`chunk_size=5000`, `overlap=200`) |

---

## Architecture

```text
                 Uploaded Files
             ┌──────────┴──────────┐
             ▼                     ▼
          PDF Files           Image Files
             │                     │
             ▼                     ▼
       PyMuPDF Loader        Gemini Vision
     (Extract Page Text)  (Extract Text Summary)
             │                     │
             └──────────┬──────────┘
                        ▼
                Merged Documents
                        ▼
            Recursive Character Splitter
              (5000 chars / 200 overlap)
                        ▼
          Gemini Embedding (text-embedding-004)
                        ▼
              Pinecone Vector Index
 ───────────────────────────────────────────────────
                  User Query
                        ▼
           Pinecone Vector Similarity Match
                        ▼
               Top-K Context Chunks
                        ▼
           LCEL Prompt + Gemini Flash LLM
                        ▼
                  Final Response
```

---

## Getting Started

### Prerequisites

- **Python 3.12+**
- **Git**
- **Google Gemini API Key**: [Get key from Google AI Studio](https://aistudio.google.com/app/apikey)
- **Pinecone API Key & Index**: [Get key from Pinecone Console](https://www.pinecone.io/)

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

Create a `.env` file in the root project directory:

```env
GOOGLE_API_KEY=your_google_gemini_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=docmind-ai
```

To run the application locally:
```bash
streamlit run app.py
```

---

## Folder Structure

```text
MultiRag_pipeline/
├── models/
│   └── vector_store.py     # Pinecone initialization, batching, and index upsert logic
├── routes/
│   └── file_router.py      # Extension-based loader routing (PDF vs Image)
├── utils/
│   ├── file_handler.py     # Local file persistence to temporary uploads directory
│   ├── chunking.py         # RecursiveCharacterTextSplitter configuration
│   ├── embeddings.py       # Initializes Gemini text-embedding-004 model
│   ├── image_loader.py     # Base64 encoding & Gemini Vision description extraction
│   ├── pdf_loader.py       # PyMuPDF text loader and page metadata extraction
│   ├── prompt.py           # System RAG ChatPromptTemplate definition
│   └── retrieval.py        # Pinecone similarity search retriever wrapper
├── uploads/                # Temporary disk storage for uploaded session files
├── app.py                  # Main Streamlit app entry point & LCEL chain execution
├── config.py               # Environment configuration loader
├── requirements.txt        # Python package dependencies
└── README.md               # Project documentation
```

---

## Performance & Design Trade-offs

- **Rate-Limit Throttling Mitigation**: Google Gemini Free Tier imposes strict limits (15 RPM). To prevent `429 Resource Exhausted` exceptions during batch indexing, vector ingestion processes in batches of 5 with controlled delay intervals and auto-retries (`max_retries=4`).
- **Image-to-Text Pipeline Trade-off**: Rather than running dual vector indices for text and images, images are translated to text descriptions via `gemini-3.1-flash-lite` first. This unifies all data into a single 768-dimensional text embedding space for low-cost, simplified search.
- **Session-State Invalidation**: Re-indexing is conditionally triggered based on sorted filename array matches (`st.session_state.processed_files`), preventing wasteful API calls during active user questioning.

---

## Current Implementation Status

- [x] Multi-file PDF parsing via PyMuPDF.
- [x] Multi-image OCR/description conversion via Gemini Vision.
- [x] Pinecone vector index initialization & similarity retrieval.
- [x] LCEL chain integration (`retriever | prompt | llm | output_parser`).
- [x] API exception handling with retry wrappers.
- [x] Stateful Streamlit UI layout.

---

## Future Roadmap & Improvements

- [ ] **Deterministic Chunk Hashing**: Replace standard Python hashing with `SHA-256` content hashing to enable true vector deduplication across sessions.
- [ ] **Multi-Tenant Namespaces**: Partition Pinecone indices using session/user namespaces to guarantee data isolation in multi-user deployments.
- [ ] **Asynchronous Processing**: Introduce Celery & Redis task queues to process large document batches without blocking UI execution.
- [ ] **Hybrid Search Integration**: Combine dense vector similarity with BM25 sparse keyword matching for improved retrieval on technical codes and product IDs.
- [ ] **In-Memory File Handling**: Transition from temporary disk writes (`uploads/`) to in-memory byte streams (`io.BytesIO`) with explicit cleanup.

---

## External API Integrations

- **Google Gemini API**: Utilized for `text-embedding-004` vector generation, `gemini-3.1-flash-lite` visual content description, and `gemini-flash-latest` RAG context answering.
- **Pinecone API**: Utilized for cloud vector storage, batched upserts, and cosine similarity query retrieval.

---

## Author

**Hardik Dhoot**
- **GitHub**: [Hardikdhoot121](https://github.com/Hardikdhoot121)

---

## License

Distributed under the MIT License. See `LICENSE` for more information.

