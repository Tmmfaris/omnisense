# OmniSense 2.0 — Hybrid Temporal Video Search

**Find exactly when something was said in a video — using natural-language and keyword search.**

OmniSense is an AI-powered video search application that transcribes video content, indexes transcript segments, and retrieves relevant moments using a combination of semantic search and keyword matching. Select a search result to generate and play the corresponding video clip.

Paste a YouTube URL or upload a video → transcribe the content → search the transcript → select a result → play the relevant segment.

## Key Features

- **Video transcription:** Convert video audio into timestamped transcript segments using `faster-whisper`.
- **Semantic search:** Retrieve relevant moments based on the meaning of a query, even when the exact words differ.
- **Keyword search:** Use BM25 to identify transcript segments containing relevant terms.
- **Hybrid ranking:** Combine semantic and keyword rankings using Reciprocal Rank Fusion (RRF).
- **Timestamp-based playback:** Select a search result to extract and play the corresponding video segment.
- **YouTube URL and file upload:** Support both input methods through the Gradio interface.
- **Search result details:** Display result timestamps, RRF scores, and semantic and keyword ranks.

## How It Works

Suppose you remember that a speaker explained deep learning but cannot remember the exact timestamp.

Search for:

`What exactly is deep learning?`

OmniSense retrieves relevant transcript segments and combines the results from two retrieval methods.

1. Semantic search finds segments with similar meaning.
2. BM25 keyword search finds segments with matching terms.
3. Reciprocal Rank Fusion combines the rankings.
4. The application displays the ranked results.
5. Selecting a result generates and plays the corresponding video clip.

### Example search results

The following illustrates the interface; the scores and results are illustrative, not measured benchmark results.

| Rank | Timestamp | Transcript |
|---|---|---|
| #1 | 00:12–00:15 | But, what exactly is deep learning? |
| #2 | 00:42–00:47 | Let's understand deep learning better and how it's different from machine learning. |
| #3 | 00:15–00:21 | Deep learning is a subset of machine learning, which in turn is a subset of artificial intelligence. |

**Why hybrid search?**

Semantic search is useful for understanding intent and paraphrases. Keyword search helps retrieve exact technical terms, names, and phrases. Combining their rankings can improve retrieval across both types of queries.

## Architecture

```text
       YouTube URL / Video Upload
                  |
                  v
          Audio Extraction
              FFmpeg
                  |
                  v
             Transcription
            faster-whisper
                  |
                  v
        Timestamped Transcript Chunks
                  |
          +-------+-------+
          |               |
          v               v
    Semantic Search   Keyword Search
      MiniLM + FAISS      BM25
          |               |
          +-------+-------+
                  |
                  v
       Reciprocal Rank Fusion
                  |
                  v
         Ranked Search Results
                  |
                  v
             Gradio UI
                  |
                  v
       Select Result and Extract
          Timestamped Video Clip
                  |
                  v
            Video Playback
```

## Technology Stack

| Category | Technologies |
|---|---|
| Programming language | Python |
| User interface | Gradio |
| Speech transcription | faster-whisper |
| Semantic embeddings | Sentence Transformers / MiniLM |
| Semantic retrieval | FAISS |
| Keyword retrieval | BM25 (`rank-bm25`) |
| Hybrid ranking | Reciprocal Rank Fusion (RRF) |
| Video and audio processing | FFmpeg |
| YouTube downloading | yt-dlp |
| Testing | pytest |
| Version control | Git and GitHub |

## Installation and Setup

### Prerequisites

- Python 3.10 or a compatible Python version supported by the project dependencies
- Git
- FFmpeg installed and accessible from your terminal

### 1. Clone the repository

```bash
git clone https://github.com/Tmmfaris/omnisense.git
cd omnisense
git checkout feature/omnisense-2.0
```

### 2. Create a virtual environment

**Windows — PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks environment activation, you can run the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**Linux / macOS**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Install FFmpeg

Install FFmpeg for your operating system and verify that the commands are available:

```bash
ffmpeg -version
ffplay -version
```

### 5. Launch the application

```bash
python -m omnisense.app
```

Open the local Gradio URL displayed in your terminal, typically:

`http://127.0.0.1:7860`

## Hybrid Search Implementation

The hybrid search pipeline combines semantic and keyword rankings.

### Semantic retrieval

The existing semantic search pipeline uses text embeddings and FAISS to retrieve transcript segments based on similarity in meaning.

### Keyword retrieval

BM25 scores transcript segments based on query-term relevance. This can help when searching for exact technical terminology or distinctive phrases.

### Reciprocal Rank Fusion

The two rankings are combined using:

\[
\operatorname{RRF}(d)=
\sum_{r \in R(d)}
\frac{1}{k+r}
\]

Where:

- \(d\) is a transcript segment.
- \(R(d)\) contains the ranks assigned to the segment by the retrieval methods.
- \(r\) is the rank of the segment in a retrieval list.
- \(k\) is a constant used to reduce the influence of higher ranks.

The combined score is used to order the results. **An RRF score is not a probability or a percentage confidence score.**

## Project Structure

```text
omnisense/
├── omnisense/
│   ├── pipelines/
│   │   ├── audio.py
│   │   ├── search.py
│   │   └── hybrid_search.py
│   ├── utils/
│   │   ├── download.py
│   │   └── logger.py
│   ├── app.py
│   └── config.py
├── tests/
│   ├── test_search.py
│   └── test_hybrid_search.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Running Tests

Run the test suite from the repository root with your virtual environment activated:

```bash
python -m pytest tests/ -v
```

The tests cover the existing semantic search pipeline and the hybrid search functionality.

## Potential Future Improvements

- Evaluate semantic-only, BM25-only, and hybrid retrieval on a labelled query dataset.
- Measure Recall@K, Precision@K, and Mean Reciprocal Rank (MRR).
- Add configurable search weights and ranking parameters.
- Improve multilingual transcript search.
- Cache transcript indexes to avoid rebuilding them unnecessarily.
- Add automated evaluation for query relevance and timestamp accuracy.
- Deploy a public demo with suitable resource limits.

These are planned improvements, not claims about features already implemented.


## Author

**Muhammed Faris**

GitHub: [@Tmmfaris](https://github.com/Tmmfaris)

---

*OmniSense 2.0 — Search video content by meaning and find the moment that matters.*
