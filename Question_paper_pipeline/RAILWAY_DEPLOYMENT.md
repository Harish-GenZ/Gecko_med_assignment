# Railway Production Deployment Guide

This guide explains how to deploy the **Gecko Med Question Paper Extraction Pipeline** to [Railway.app](https://railway.app).

---

## Architecture Options

You can deploy the pipeline to Railway using either of two battle-tested production architectures:

| Approach | Setup | Best For | Memory / Costs |
| :--- | :--- | :--- | :--- |
| **Option A: Unified Full-Stack Container** *(Recommended)* | 1 Railway Service | 1-Click Deploy, lowest resource usage, zero CORS configuration | $\sim 1.5 - 2.5\text{ GB}$ RAM |
| **Option B: Separate Microservices** | 2 Railway Services (Backend + Frontend) | Independent scaling of UI and OCR pipeline | $\sim 2.5 - 3.5\text{ GB}$ RAM |

---

## Option A: Unified Full-Stack Container (Recommended)

In this approach, Railway builds the React frontend and packages it inside the FastAPI backend. FastAPI serves the React UI at root (`/`) and the API at (`/api`), eliminating CORS issues and requiring only a single Railway service.

### Step-by-Step Deployment:

1. **Push Code to GitHub**:
   Ensure all changes are committed and pushed to your GitHub repository.

2. **Create New Project on Railway**:
   - Log into [Railway.app](https://railway.app).
   - Click **New Project** $\to$ **Deploy from GitHub repo**.
   - Select your repository.
   - If prompted for a root directory, select `Question_paper_pipeline` (or leave default if the repository root is the pipeline).

3. **Configure Railway Environment Variables**:
   In your Railway service settings $\to$ **Variables**, add:
   ```env
   GEMINI_API_KEY=your_google_gemini_api_key_here
   GEMINI_VISION_MODEL=gemini-3.1-flash-lite
   OCR_WORKERS=4
   VISION_MAX_CONCURRENT=4
   VISION_ENABLED=true
   VISION_TIMEOUT_SECONDS=30
   JOB_TIMEOUT=2400
   ```
   *(Note: Railway automatically provides and assigns the `PORT` environment variable).*

4. **Add Persistent Storage (Volume)** *(Optional but Recommended)*:
   Railway containers have ephemeral disks by default. To preserve uploaded PDFs and JSON extraction jobs across redeployments:
   - Click on your Railway service $\to$ **Volumes** $\to$ **Add Volume**.
   - Set the mount path to:
     ```
     /app/storage
     ```

5. **Generate Public Domain**:
   - Under service settings $\to$ **Networking**, click **Generate Domain**.
   - Open your generated domain (e.g. `https://your-service.up.railway.app`).
   - The React UI will load instantly, with the `/api` backend already connected!

---

## Option B: Separate Microservices

If you prefer deploying the Backend and Frontend as two independent Railway services:

### 1. Backend Service:
- Create service from GitHub repo $\to$ set Root Directory to `Question_paper_pipeline/backend`.
- Builder: **Dockerfile** (`backend/Dockerfile`).
- Add environment variables:
  ```env
  GEMINI_API_KEY=your_gemini_api_key
  GEMINI_VISION_MODEL=gemini-3.1-flash-lite
  OCR_WORKERS=4
  VISION_MAX_CONCURRENT=4
  CORS_ORIGINS=*
  ```
- Generate a domain (e.g. `https://gecko-backend.up.railway.app`).
- Health Check Path: `/api/health`.

### 2. Frontend Service:
- Add a second service from the same repo $\to$ set Root Directory to `Question_paper_pipeline/frontend`.
- Builder: **Dockerfile** (`frontend/Dockerfile.frontend`).
- Add environment variable:
  ```env
  VITE_API_URL=https://gecko-backend.up.railway.app
  ```
- Generate domain for the frontend.

---

## Health Check & Verification

Once deployed, verify your deployment:

1. **Health Check Endpoint**:
   ```bash
   curl -I https://your-domain.up.railway.app/api/health
   # Returns: HTTP/1.1 200 OK {"status": "ok"}
   ```

2. **Interactive API Documentation (Swagger)**:
   Navigate to:
   ```
   https://your-domain.up.railway.app/docs
   ```

3. **Production Best Practices Applied**:
   - `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1` enforced to prevent OpenBLAS thread crashes on cloud Linux kernels.
   - PaddleOCR models pre-baked into container image to eliminate Baidu mirror cold-start download timeouts.
   - Headless OpenCV (`opencv-python-headless`) to avoid missing X11/GL libraries on Linux.
   - Dynamic port binding `${PORT:-8000}` compatible with Railway runtime.
   - Fault-tolerant sliding window task dispatch with sequential fallback protection.
