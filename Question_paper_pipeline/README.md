# Gecko Med — Question Paper Extraction Pipeline

An AI/ML pipeline that extracts and classifies questions from medical exam PDFs using OCR (PaddleOCR) and a FastAPI backend with a React/TypeScript frontend.

---

## Project Structure

```
Question_paper_pipeline/
├── backend/               # FastAPI Python backend
│   ├── app/
│   │   ├── api/           # Route handlers
│   │   ├── core/          # Job store, config
│   │   ├── models/        # Pydantic schemas
│   │   └── services/      # OCR engine, extraction logic
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/              # React + TypeScript (Vite) frontend
│   ├── src/
│   │   ├── App.tsx
│   │   ├── types.ts
│   │   └── services/api.ts
│   └── index.html
├── docker-compose.yml
└── README.md
```

---

## Prerequisites

| Tool | Minimum Version |
|------|----------------|
| Python | 3.10+ |
| Node.js | 18+ |
| npm | 9+ |
| Docker & Docker Compose | (optional, for containerised run) |

---

## Running the App

### Option 1 — Local Development (Recommended)

#### 1. Start the Backend

```bash
# From the project root
cd Question_paper_pipeline\backend

# Create a virtual environment
python -m venv .venv

# Activate it (Windows PowerShell)
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server — run from the backend/ folder, NOT backend/app/
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at **http://localhost:8000**  
Interactive docs: **http://localhost:8000/docs**

#### 2. Start the Frontend

Open a **new terminal** from the project root:

```bash
cd frontend

# Install Node dependencies (first time only)
npm install

# Start the Vite dev server
npm run dev
```

The React app will be available at **http://localhost:5173**

> The Vite dev server proxies `/api` requests to `http://localhost:8000` automatically.

---

### Option 2 — Docker Compose

Make sure Docker Desktop is running, then from the project root:

```bash
docker compose up --build
```

| Service  | URL                        |
|----------|----------------------------|
| Frontend | http://localhost:5173      |
| Backend  | http://localhost:8000      |
| API Docs | http://localhost:8000/docs |

To stop the containers:

```bash
docker compose down
```

---

## Usage

1. Open **http://localhost:5173** in your browser.
2. Upload a medical exam PDF using the upload button.
3. The pipeline processes the document (OCR for scanned pages, text extraction for digital pages).
4. View the extracted questions, their types (MCQ / Essay / Short Notes / etc.), marks, and confidence scores.
5. Questions flagged with `needs_review` status are highlighted for manual verification.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/upload` | Upload a PDF and start extraction job |
| `GET` | `/api/job/{job_id}` | Poll job status and retrieve results |
| `GET` | `/api/health` | Health check |
| `GET` | `/docs` | Swagger UI |

---

## Tech Stack

**Backend**
- [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) for scanned page OCR
- [PyMuPDF](https://pymupdf.readthedocs.io/) for PDF text extraction
- Pydantic v2 for data validation

**Frontend**
- [React 18](https://react.dev/) + TypeScript
- [Vite](https://vitejs.dev/) dev server
- Vanilla CSS
