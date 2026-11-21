# VerifAI · Evidence-Grounded Biomedical Q&A

VerifAI answers clinical and biomedical questions **only from curated sources**, cites the passage behind every claim, verifies each answer against its evidence, and measures itself against a classic keyword-search baseline.

- **Conversational:** greetings, follow-up questions that keep context ("how is it treated?"), answers streamed word by word
- **Grounded:** answers come only from the built-in knowledge base (120 topics) and datasets you upload; uncovered questions are declined instead of guessed
- **Verified:** every answer gets a verification status, cited sources and a measured answer report
- **Measured:** live benchmark of VerifAI (BM25 + LLM) vs a TF-IDF baseline on 380 labelled questions
- **Private by default:** runs fully on your machine with a free local model through [Ollama](https://ollama.com); OpenAI is optional

> For information and research only. Not a substitute for professional medical advice.

---

## Tech stack

| Layer | Technology |
|---|---|
| Web | FastAPI, Uvicorn, Jinja2 templates |
| Data | SQLAlchemy 2 + SQLite |
| Retrieval | BM25 (VerifAI) and TF-IDF / scikit-learn (baseline) |
| Answers | Local LLM via Ollama (`llama3.2:3b` by default) or OpenAI (`gpt-4o-mini`) |
| Frontend | Bootstrap 5, Chart.js, tsParticles, GSAP, Typed.js, CountUp.js |

---

## Run it locally

### 1. Prerequisites

- **Python 3.12** (3.11 also works; 3.13+ is not supported by the pinned scientific packages)
- **Git**
- **Ollama** for the free local AI model: download from <https://ollama.com/download>
  (optional if you use an OpenAI API key instead)
- About **4 GB of free RAM** for the local model

### 2. Get the code

```bash
git clone https://github.com/MYethishwar/verifai-evidence-grounded-qa.git
cd verifai-evidence-grounded-qa
```

### 3. Create a virtual environment and install dependencies

**Windows (PowerShell)**
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux**
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Download the local AI model (one time)

With Ollama installed and running:

```bash
ollama pull llama3.2:3b
```

Skip this step if you use OpenAI instead; see [Configuration](#configuration).

### 5. Start the app

```bash
python app.py
```

Open **<http://127.0.0.1:5000>**. Interactive API docs are at **<http://127.0.0.1:5000/docs>**.

On first start the app creates its database (`instance/verifai.db`), an administrator account, sample biomedical articles and the search index. The first answer can take a little longer while the model loads.

### 6. Sign in

| Role | How |
|---|---|
| Administrator | Username `admin`, password `admin123` (change these in `config.py` before sharing the app) |
| User | Click **Request access** on the sign-in page, then approve the account as the administrator under **Users & access** |

---

## Using VerifAI

- **Ask VerifAI:** chat with the assistant. Try *"What does the HbA1c test measure?"*, then follow up with *"Do I need to fast for it?"*
- **Knowledge base:** browse all 120 topics (conditions, medicines, lab tests, biology basics) and click any sample question
- **Answer report:** every answer shows its sources, verification status and measured metrics next to the baseline
- **Benchmark:** labelled retrieval benchmark (Hit@1, Recall@5, MRR, nDCG, latency) for both systems
- **Admin:** upload datasets (CSV, Excel, JSON, TXT), rebuild the index, manage users, and review quality metrics

Add new domain knowledge by editing or adding JSON files in `knowledge_base/`; no code changes are needed.

---

## Configuration

Set these environment variables before `python app.py` (all optional):

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_MODEL` | `llama3.2:3b` | Local model to use (e.g. `llama3.2:1b` is faster but less precise) |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server address |
| `OLLAMA_NUM_CTX` | `2048` | Model context window (smaller uses less memory) |
| `OPENAI_API_KEY` | (empty) | Use OpenAI instead of the local model |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model name |
| `VERIFAI_ALLOW_GENERAL_KNOWLEDGE` | `0` | Set to `1` to let the model answer uncovered questions from general knowledge (clearly labelled) |
| `VERIFAI_DATABASE_URL` | `sqlite:///instance/verifai.db` | Database location |
| `SECRET_KEY` | built-in default | Session signing key; **set your own for any shared deployment** |

PowerShell example:
```powershell
$env:OLLAMA_MODEL = "llama3.2:1b"
python app.py
```

---

## Project structure

```
app.py                  FastAPI application, startup and seed data
web.py                  Templates, sessions, flash messages, sign-in dependencies
extensions.py           SQLAlchemy setup (one session per request)
config.py               Settings
routes/                 auth, user (chat, reports, benchmark), admin, api
services/
  retrieval_service.py  BM25 + TF-IDF search with topic-aware filtering
  ai_service.py         Answer planning, grounded generation, citations
  chat_service.py       Conversation flow, follow-ups, small talk
  domain_guard.py       Biomedical scope check
  verification_service.py  Answer verification against sources
  evaluation_service.py Measured per-answer reports and labelled benchmark
  knowledge_base.py     Loads the curated knowledge base
  llm_client.py         Ollama / OpenAI client (streaming)
knowledge_base/         120 curated topics as JSON
datasets/               Sample datasets and the labelled test questions
templates/, static/     Pages, styles, scripts, logo
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `pip install` fails on numpy/pandas | Use Python 3.12 (see Prerequisites) |
| Header shows "AI model offline · rule-based answers" | Start Ollama and run `ollama pull llama3.2:3b`, then restart the app |
| Answers are slow | The local model runs on the CPU; close memory-heavy apps, or try `OLLAMA_MODEL=llama3.2:1b` |
| `ImportError: DLL load failed ... Application Control policy` (Windows) | Windows Smart App Control blocked a compiled package; `requirements.txt` already pins SQLAlchemy 2.0.x, which falls back to pure Python |
| Want a clean start | Stop the app and delete the `instance/` folder; it is recreated on the next start |
