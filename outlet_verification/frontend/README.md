# Outlet Verification Frontend

A clean, responsive, operations-grade web interface for testing and interacting with the Outlet Verification Engine.

---

## 1. Prerequisites

- **Node.js**: `v18+` (Tested on `v22.19.0`)
- **npm**: `v9+` (Tested on `v10.9.3`)
- **Python**: `3.10+` with the `outlet_verification` virtual environment configured

---

## 2. Installation

From the `outlet_verification/frontend` directory:

```bash
npm install
```

---

## 3. Environment Variables

Create a `.env` file in `outlet_verification/frontend/`:

```env
# URL where the FastAPI backend is running
VITE_API_BASE_URL=http://localhost:8000
```

An example template is available at `.env.example`.

---

## 4. How to Start the Frontend

Development server:

```bash
npm run dev
```

The frontend will be available at:

```
http://localhost:5173
```

---

## 5. How to Start the Backend

From the `outlet_verification/` root directory:

```powershell
.\venv\Scripts\uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Backend will be available at:
- Root: `http://localhost:8000`
- Health check: `http://localhost:8000/health`
- Interactive Swagger docs: `http://localhost:8000/docs`

---

## 6. API Endpoints Used

| Endpoint | Method | Payload | Purpose |
| :--- | :---: | :--- | :--- |
| `/health` | `GET` | None | Real-time backend status and PostgreSQL connection indicator |
| `/verify` | `POST` | `multipart/form-data`<br>- `name`: string<br>- `latitude`: float<br>- `longitude`: float<br>- `image`: file | Executes full multi-channel candidate retrieval, multimodal evidence fusion, and returns deterministic decision (`DUPLICATE`, `GENUINE`, or `NEEDS_REVIEW`) |

---

## 7. Manual Testing Workflow

1. **Verify Backend Status**: Look at the top-right status pill in the header. It should display `Backend: Connected` with an emerald indicator.
2. **Select or Enter Outlet Details**:
   - Click one of the **Quick-Fill Sample Outlets** preset buttons (e.g., `Indiranagar (Bangalore)`) to automatically populate known reference coordinates and names, OR type a custom name and coordinates.
   - Click **Use Current Location** to test browser geolocation API.
3. **Upload Storefront Photograph**:
   - Drag and drop or browse to select an image (`.jpg`, `.jpeg`, `.png`, `.webp`).
   - Notice the client-side metadata display (filename, size, dimensions) and preview.
4. **Click "Verify Outlet with AI Engine"**:
   - View the active loading state illustrating the stages (embedding generation, candidate retrieval, database query, multimodal fusion).
5. **Inspect Verification Results**:
   - **Decision Banner**: Shows `DUPLICATE`, `GENUINE`, or `NEEDS_REVIEW` with exact engineered confidence score.
   - **Multimodal Evidence Profile**: Displays name similarity, visual image similarity, GPS distance, inter-signal agreement, and candidate margin.
   - **Visual Comparison**: Side-by-side inspection of the submitted photograph vs. the matched database reference photo.
   - **Audit Reason Codes**: Review deterministic decision codes (e.g. `STRONG_MULTIMODAL_MATCH`, `AMBIGUOUS_TOP_CANDIDATES`).
6. **Reset**: Click **Verify Another Outlet** to return to a clean state.

---

## 8. Troubleshooting CORS

The FastAPI backend in `app/main.py` is configured with `CORSMiddleware`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

If you encounter `NetworkError` or `CORS policy` blocks in your browser console:
1. Verify the FastAPI backend is running on `http://localhost:8000`.
2. Verify `VITE_API_BASE_URL` in `frontend/.env` matches the backend host and port.
3. Test `curl http://localhost:8000/health` or open `http://localhost:8000/health` in your browser.

---

## 9. Production Build

To compile TypeScript and bundle the frontend for production deployment:

```bash
npm run build
```

Production static assets will be output to `frontend/dist/`.

To preview the production build locally:

```bash
npm run preview
```
