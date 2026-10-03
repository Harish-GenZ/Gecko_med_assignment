# Railway Production Deployment Guide (v2 Neural Network Pipeline)

This guide walks you through deploying the **Gecko Med Question Paper Extraction Pipeline (v2 NN)** to [Railway.app](https://railway.app).

---

## Architecture: Unified Full-Stack Container (Recommended)

In this setup, Railway builds the React frontend and packages it directly inside the FastAPI backend container using multi-stage Docker. FastAPI serves the compiled React SPA at `/` and the neural extraction API at `/api`, providing:
- **1-Click Deployment** (only 1 Railway service required).
- **Zero CORS configuration** (Frontend and Backend share the same domain and port).
- **Lowest resource consumption & costs** (well within Railway's Hobby / Pro tiers).
- **Zero Cold-Start Latency** (RapidOCR ONNX models are pre-cached in the Docker image).

---

## 1-Click Deployment Instructions

### Step 1: Push Code to GitHub
Ensure all code in `Question_paper_pipeline_v2_NN` is pushed to your GitHub repository.

### Step 2: Create a New Project on Railway
1. Log into [Railway.app](https://railway.app).
2. Click **New Project** $\to$ **Deploy from GitHub repo**.
3. Select your repository.
4. Set the **Root Directory** to `Question_paper_pipeline_v2_NN` (or leave as `/` if your repository root is this folder).
5. Railway will automatically detect `railway.json` and `Dockerfile`.

### Step 3: Configure Environment Variables
In your Railway dashboard $\to$ **Variables**, add:

| Variable | Recommended Value | Description |
| :--- | :--- | :--- |
| `OCR_WORKERS` | `2` | Number of worker processes (2 is optimal for 1–2 vCPU Railway containers) |
| `VISION_ENABLED` | `true` | Enables asynchronous vision verification for flagged pages |
| `VISION_MAX_CONCURRENT` | `2` | Max concurrent vision verification workers |
| `GEMINI_API_KEY` | *(Your key or leave empty)* | Optional Google Gemini key; if empty, Local VLM handles verification |
| `MAX_FILE_SIZE_MB` | `200` | Max PDF file size allowed |
| `JOB_TIMEOUT` | `2400` | Processing timeout in seconds (40 mins for large 50+ page PDFs) |

*(Note: Railway automatically provides and assigns the `PORT` environment variable. The Dockerfile binds to `${PORT:-8000}` automatically).*

### Step 4: Add Persistent Storage Volume *(Recommended)*
Railway containers have ephemeral disks by default. To preserve uploaded PDFs and JSON extraction jobs across redeployments:
1. In your Railway service view, click **Volumes** $\to$ **Add Volume**.
2. Set the **Mount Path** to:
   ```
   /app/storage
   ```
3. Click **Save**. All uploads and extracted JSON results will persist across container restarts.

### Step 5: Generate Public Domain
1. In your service settings $\to$ **Networking**, click **Generate Domain**.
2. Open your generated domain (e.g. `https://your-pipeline.up.railway.app`).
3. The React web app will load immediately, fully connected to the neural extraction API!

---

## Verification & Health Check

1. **Health Check Endpoint**:
   ```bash
   curl -I https://your-pipeline.up.railway.app/api/health
   # Returns: HTTP/1.1 200 OK {"status": "ok"}
   ```

2. **Interactive OpenAPI / Swagger Documentation**:
   Navigate to:
   ```
   https://your-pipeline.up.railway.app/docs
   ```

---

## Production Optimizations Included in Dockerfile

- **CPU-Optimized PyTorch**: Pre-installs PyTorch CPU wheels (`torchvision==cpu`), avoiding the 2.8 GB NVIDIA/CUDA bloat and reducing build time from 15 minutes to under 2 minutes.
- **Headless OpenCV**: Uses `opencv-python-headless` to eliminate missing X11/GL system dependencies on Debian Linux.
- **Pre-Cached Neural Weights**: RapidOCR ONNX models are downloaded and baked into the image layer at build time, preventing runtime download delays or connection timeouts.
- **Multi-Thread CPU Safety**: Enforces `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1` to prevent OpenBLAS thread collisions on cloud Linux kernels.
- **Dynamic Port Binding**: Conforms to `${PORT:-8000}` matching Railway's dynamic ingress routing.
