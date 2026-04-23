# Moroccan Legal AI Assistant

Professional legal consultation and document intelligence for Moroccan law, powered by a Retrieval-Augmented Generation (RAG) pipeline.

## Table of Contents

- [Overview](#overview)
- [Current Architecture](#current-architecture)
- [Requirements](#requirements)
- [Setup](#setup)
- [Run the Project](#run-the-project)
- [Project Structure](#project-structure)
- [Configuration Notes](#configuration-notes)
- [Troubleshooting](#troubleshooting)
- [Security and Legal Notice](#security-and-legal-notice)

## Overview

The assistant is specialized in Moroccan legal domains (family, criminal, civil, constitutional law) and supports French and Arabic queries.

Core capabilities:

- Multi-backend LLM fallback (OpenRouter, Groq, OpenAI)
- Hybrid legal retrieval over ChromaDB
- PDF legal document analysis
- Article extraction and source citation
- Bilingual legal interaction

## Current Architecture

The project now uses a separated frontend/backend stack:

- Frontend: React + Vite (`frontend/`)
- Backend API: FastAPI (`legal_api_server.py`)
- Core legal engine: Python RAG pipeline (`moroccan_legal_chatbot.py`)

Flow:

1. User interacts with React UI (`localhost:5173`)
2. Frontend calls FastAPI endpoints (`localhost:8000`)
3. API delegates to the legal engine
4. Legal engine performs retrieval + generation and returns response

Legacy mode is still available:

- `python moroccan_legal_chatbot.py` launches the integrated Gradio interface.

## Requirements

### Backend

- Python 3.10+ recommended
- A virtual environment
- At least one API key:
  - `OPENROUTER_API_KEY`
  - `GROQ_API_KEY`
  - `OPENAI_API_KEY`

### Frontend

- Node.js LTS (includes npm)
- npm 10+ recommended

## Setup

### 1) Clone and enter project

```bash
git clone https://github.com/symooomzip/moroccan-legal-chatbot.git
cd moroccan-legal-chatbot
```

### 2) Create and activate virtual environment

```bash
python -m venv .venv
```

Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

Windows (CMD):

```bat
.\.venv\Scripts\activate
```

Linux/Mac:

```bash
source .venv/bin/activate
```

### 3) Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4) Configure `.env`

Create `.env` in project root:

```env
OPENROUTER_API_KEY=your_openrouter_key_here
GROQ_API_KEY=your_groq_key_here
OPENAI_API_KEY=your_openai_key_here
```

### 5) Install frontend dependencies

```bash
cd frontend
npm install
```

If npm is not recognized in your current terminal on Windows:

```bat
set "PATH=C:\Program Files\nodejs;%PATH%"
"C:\Program Files\nodejs\npm.cmd" install
```

## Run the Project

You need two terminals.

### Terminal A - Backend API

From project root:

```powershell
cd "D:\master\School\S9\Application avancees en Intelligence Artificielle\Chatbot_prj"
.\.venv\Scripts\Activate.ps1
python -m uvicorn legal_api_server:app --host 0.0.0.0 --port 8000 --reload
```

### Terminal B - Frontend

From `frontend/`:

```powershell
cd "D:\master\School\S9\Application avancees en Intelligence Artificielle\Chatbot_prj\frontend"
npm run dev
```

Open:

- Frontend UI: [http://localhost:5173](http://localhost:5173)
- API health: [http://localhost:8000/health](http://localhost:8000/health)

## Project Structure

```text
Chatbot_prj/
├── moroccan_legal_chatbot.py      # Core legal engine + legacy Gradio UI
├── legal_api_server.py            # FastAPI wrapper for React frontend
├── requirements.txt
├── .env
├── README.md
├── frontend/
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── App.jsx
│       ├── api.js
│       ├── main.jsx
│       └── styles.css
├── legal_docs/                    # PDF corpus grouped by legal domain
├── chroma_db/                     # Generated vector database
└── user_uploads/                  # Temporary uploaded documents
```

## Configuration Notes

Main runtime configuration lives in `moroccan_legal_chatbot.py` under `Config`.

Important settings:

- `EMBEDDING_MODEL`
- `CHROMA_DB_PATH`
- `LEGAL_DOCS_PATH`
- `TOP_K_RESULTS`
- `TIMEOUT_SECONDS`

For constrained machines, keep this environment setting:

```env
TOKENIZERS_PARALLELISM=false
```

## Troubleshooting

### `npm` not recognized

1. Install Node.js LTS from [nodejs.org](https://nodejs.org)
2. Fully restart terminal/Cursor
3. Verify:

```bat
node -v
npm -v
```

### `npm run dev` fails with `ENOENT package.json`

You are running the command from project root. Move to `frontend/` first:

```bat
cd frontend
npm run dev
```

### `npm run dev` fails with `"node" is not recognized`

Current terminal does not include Node in PATH. Fix this session:

```bat
set "PATH=C:\Program Files\nodejs;%PATH%"
npm run dev
```

### Backend memory allocation error at startup

If you see errors like `memory allocation ... bytes failed`:

- Close heavy applications
- Increase Windows page file/virtual memory
- Keep `TOKENIZERS_PARALLELISM=false`
- Retry backend start

### API starts but frontend cannot connect

- Confirm backend is running on port 8000
- Confirm frontend is running on port 5173
- Open [http://localhost:8000/health](http://localhost:8000/health)

## Security and Legal Notice

- Store API keys only in `.env` and never commit secrets
- Validate generated outputs with a qualified legal professional
- This system provides legal information support, not legal representation

---

This project is part of an academic AI application workflow and is continuously improved for stability, UI quality, and legal retrieval accuracy.

## 👥 Authors

*   **Mohammed Fakir** - [GitHub: @Symooomzip](https://github.com/Symooomzip)
*   **Lubabah Hamouch** - [GitHub: @Lubabah-Hamouch](https://github.com/Lubabah-Hamouch)
