# Outlet Verification System - Backend

## 1. Project Purpose
The Outlet Verification System is designed for a field-sales application where sales representatives register outlets by submitting:
1. Outlet Name
2. Latitude & Longitude
3. Outlet Image

The system's goal is to determine:
- Whether an outlet submission is **genuine** or fake.
- Whether it is a **duplicate** of an already registered outlet (even with altered names, photos, or slightly different GPS coordinates).
- Whether the available information is **insufficient**, requiring manual human review.

---

## 2. Environment Setup

### Prerequisites
- Python 3.10+ (Tested on Python 3.13)
- PowerShell or bash

### Creating the Virtual Environment
From inside the `outlet_verification/` folder:

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Windows (CMD):
.\venv\Scripts\activate.bat
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 3. Configuration & Environment Variables

Create a `.env` file in `outlet_verification/` based on `.env.example`:

```env
APP_NAME="Outlet Verification System"
APP_ENV=development
DEBUG=True
HOST=0.0.0.0
PORT=8000

# Railway PostgreSQL Connection URL
DATABASE_URL=postgresql://postgres:<password>@<host>:<port>/railway

# Retrieval Parameters (Phase 4)
RETRIEVAL_NAME_TOP_K=5
RETRIEVAL_IMAGE_TOP_K=5
RETRIEVAL_GEO_RADIUS_METERS=500.0
RETRIEVAL_GEO_TOP_K=10
```

> **Important Note regarding Railway URLs:**
> - Railway provides two URLs:
>   1. **Private URL** (`postgres.railway.internal`): Only accessible by services running inside Railway's cloud network.
>   2. **Public URL** (e.g., `junction.proxy.rlwy.net:12345` or `postgres.railway.app:5432`): Accessible by external machines and local development.
> - When running locally outside Railway, ensure you use the **Public TCP Proxy URL** provided in Railway under **PostgreSQL > Settings / Connect > Public Networking**.

---

## 4. Running the Backend

```powershell
# Using uvicorn with the virtual environment activated:
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Or directly running main.py:
python -m app.main
```

The interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## 5. Health Endpoints & Diagnostics

### Basic Health Check (`GET /health`)
Verifies that the backend process is running and checks Railway PostgreSQL connectivity:

```powershell
curl http://localhost:8000/health
```

### Detailed Diagnostics (`GET /health/db`)
Returns PostgreSQL version details, `pgvector` extension availability, and installation status:

```powershell
curl http://localhost:8000/health/db
```

---

## 6. Railway Connection & pgvector Initialization Mechanism

The system includes a safe, idempotent database verification mechanism (`app/db/init_db.py` and `scripts/verify_db.py`):

1. **Connection Test**: Executes `SELECT 1;`.
2. **Version Inspection**: Executes `SELECT version();` to retrieve PostgreSQL release details.
3. **Extension Inspection**: Queries `pg_available_extensions` for `name = 'vector'`.
4. **Idempotent Activation**: If available, executes:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```
5. **Verification**: Queries `pg_extension` to confirm activation and retrieve installed version.

You can run the standalone CLI verification anytime:

```powershell
python scripts/verify_db.py
```

---

## 7. Phase 3B: Embedding Infrastructure

### Selected Baseline Models
- **Text Embedding Model**: `all-MiniLM-L6-v2` (Sentence-Transformers)
  - **Vector Dimension**: `384`
  - **Purpose**: Outlet name and entity matching, accommodating variations in brand spelling, word ordering, and abbreviations.
  - **Why Selected**: Ultra-fast CPU inference (~10-15 ms), lightweight footprint (~90 MB), Apache 2.0 license.
- **Image Embedding Model**: `CLIP ViT-B/32` (`openai/clip-vit-base-patch32`)
  - **Vector Dimension**: `512`
  - **Purpose**: Storefront visual similarity for duplicate photo detection and future cross-modal verification.
  - **Why Selected**: MIT license, standard representation, efficient CPU performance with 32x32 patches.

> **Note on Evaluation**: These models serve as **baseline models** that will be empirically evaluated and calibrated against our project dataset in subsequent phases. They are not claimed to be definitive.
>
> **Important Note on GPS**: Coordinates (Latitude and Longitude) are **NOT converted into vector embeddings**. Geospatial proximity is handled using physical earth distances (`earthdistance` / PostGIS geographic distance) to preserve exact metric physical distances in meters.

### Model Loading & Lifecycle
- Models are instantiated as **thread-safe singletons** (`TextEmbeddingService`, `ImageEmbeddingService`).
- Models are initialized **once per application worker** and reused across all incoming requests.
- No model reloading or repeated instantiation occurs per request.
- All operations run purely on **CPU** without requiring GPU acceleration.

### Vector Storage in PostgreSQL
Columns in table `outlets`:
```sql
ALTER TABLE outlets ADD COLUMN name_embedding vector(384) NULL;
ALTER TABLE outlets ADD COLUMN image_embedding vector(512) NULL;
```
- Existing rows are preserved with `NULL` embeddings.
- SQLAlchemy ORM maps `name_embedding` and `image_embedding` using `pgvector.sqlalchemy.Vector` (with an automatic fallback type).

---

## 8. Phase 4: Candidate Retrieval & Spatial Pre-filtering

### Architecture Overview
Phase 4 implements a multi-channel candidate retrieval layer. Given an outlet submission (name, GPS coordinates, and storefront photo), it surfaces existing database outlets that could plausibly represent the same physical outlet.

The retrieval layer queries three distinct channels directly inside PostgreSQL:
1. **Name Vector Retrieval**: Semantic search using pgvector HNSW index with cosine distance on `name_embedding` (`<=>` operator).
2. **Image Vector Retrieval**: Visual search using pgvector HNSW index with cosine distance on `image_embedding` (`<=>` operator).
3. **Geographic Retrieval**: Spatial pre-filtering using PostgreSQL `earthdistance` extension (`earth_distance(ll_to_earth(...)) <= radius_meters`) backed by a GiST index on `ll_to_earth(latitude, longitude)`.

```
                    ┌───────────────────────────────┐
                    │     New Outlet Submission     │
                    │  (Name, Coordinates, Photo)   │
                    └───────────────┬───────────────┘
                                    │
               ┌────────────────────┼────────────────────┐
               ▼                    ▼                    ▼
     ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
     │   Name Vector    │  │   Image Vector   │  │   Geographic     │
     │      Search      │  │      Search      │  │     Radius       │
     │   (HNSW 384d)    │  │   (HNSW 512d)    │  │  (GiST Spatial)  │
     └─────────┬────────┘  └─────────┬────────┘  └─────────┬────────┘
               │                     │                     │
               └────────────────────┬┴─────────────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │ Candidate Aggregation & Merge │
                    │   (Deduplicated by Outlet ID) │
                    └───────────────┬───────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │        Candidate Pool         │
                    │   (Evidence Profile Output)   │
                    └───────────────────────────────┘
```

### PostgreSQL Indexes (Alembic Migration 003)
```sql
-- HNSW Cosine Index for Name Embeddings
CREATE INDEX idx_outlets_name_embedding_hnsw 
ON outlets USING hnsw (name_embedding vector_cosine_ops);

-- HNSW Cosine Index for Image Embeddings
CREATE INDEX idx_outlets_image_embedding_hnsw 
ON outlets USING hnsw (image_embedding vector_cosine_ops);

-- GiST Spatial Index for Latitude/Longitude Earth Distance
CREATE INDEX idx_outlets_ll_to_earth 
ON outlets USING gist (ll_to_earth(latitude, longitude));
```

### Initial Retrieval Parameters
Configured in `app/config.py` (and overridable via `.env` or per-request):
- `RETRIEVAL_NAME_TOP_K`: `5` (Top candidates from name semantic search)
- `RETRIEVAL_IMAGE_TOP_K`: `5` (Top candidates from image visual search)
- `RETRIEVAL_GEO_RADIUS_METERS`: `500.0` (Pre-filtering search radius in meters)
- `RETRIEVAL_GEO_TOP_K`: `10` (Maximum candidates from geographic radius)

> **Important Distinction**: Candidate retrieval **$\neq$ duplicate decision**. 
> The candidate retrieval layer answers: *"Which existing outlets are plausible candidates for comparison?"*
> It does NOT assign confidence scores, genuine/fake labels, or duplicate verdicts. Those belong to later phases.

### Development Testing Endpoints
- `POST /candidates/search`: Accepts multipart form (`name`, `latitude`, `longitude`, `image` file).
- `POST /candidates/search-json`: Accepts JSON payload (`name`, `latitude`, `longitude`, `image_base64` or `image_url`).

### Running Phase 4 Tests & Demo
```powershell
# Run the candidate retrieval automated test suite
python tests/test_candidate_retrieval.py

# Run the live candidate retrieval demonstration script
python scripts/verify_candidates.py
```

---

## 8. Phase 5 — Multimodal Fusion & Verification Decision Engine

### Overview
Phase 5 implements the deterministic multimodal evidence fusion and decision engine. Given a submitted outlet (`name`, `latitude`, `longitude`, `image`), the system leverages Phase 4 retrieval to surface candidates, normalizes multi-modal signals, calculates signal consensus and evidence coverage, and outputs a deterministic decision (`DUPLICATE`, `GENUINE`, or `NEEDS_REVIEW`) with structured reason codes.

```
                    ┌───────────────────────────────┐
                    │     New Outlet Submission     │
                    │  (Name, Coordinates, Photo)   │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │ Candidate Retrieval (Phase 4) │
                    │   (Name, Image, Geo Search)   │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │  Evidence Normalization & G   │
                    │   G = exp(-distance / D)      │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   Candidate Scoring Engine    │
                    │   - Weight Renormalization    │
                    │   - Evidence Coverage         │
                    │   - Signal Agreement          │
                    │   - Duplicate Confidence      │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │ Deterministic Decision Engine │
                    │   - Conflict Safeguards       │
                    │   - Candidate Margin Ambiguity│
                    │   - Decision Thresholds       │
                    └───────────────┬───────────────┘
                                    │
             ┌──────────────────────┼──────────────────────┐
             ▼                      ▼                      ▼
     ┌───────────────┐      ┌───────────────┐      ┌───────────────┐
     │   DUPLICATE   │      │    GENUINE    │      │ NEEDS_REVIEW  │
     └───────────────┘      └───────────────┘      └───────────────┘
```

### Mathematical Formulation

1. **Geographic Proximity Score ($G$)**:
   Distance in meters ($d$) is transformed into a continuous proximity score $G \in [0.0, 1.0]$ using exponential decay:
   $$G = \exp\left(-\frac{d}{D}\right)$$
   where $D$ is `VERIFICATION_GEO_SCALE_METERS` (default: $100.0\text{ m}$).
   - $d = 0\text{ m} \implies G = 1.0$
   - $d = 25\text{ m} \implies G \approx 0.779$
   - $d = 50\text{ m} \implies G \approx 0.607$
   - $d = 100\text{ m} \implies G \approx 0.368$
   - $d = 500\text{ m} \implies G \approx 0.0067$

2. **Base Multimodal Score ($S_{\text{base}}$) with Weight Renormalization**:
   Missing signals (e.g. image unavailable or GPS missing) are treated as *unknown*, **never as zero**. The available signal weights are dynamically renormalized:
   $$S_{\text{base}} = \frac{\sum_{i \in \text{available}} w_i \cdot s_i}{\sum_{i \in \text{available}} w_i}$$
   Initial configured weights:
   - $w_{\text{name}} = 0.30$
   - $w_{\text{image}} = 0.45$
   - $w_{\text{geo}} = 0.25$
   *(Validated at startup: $\sum w_i = 1.0$)*

3. **Evidence Coverage Metric**:
   Indicates the proportion of intended evidence actually available:
   $$\text{coverage} = \frac{\sum_{i \in \text{available}} w_i}{\sum_{\text{all}} w_i} \in [0.0, 1.0]$$

4. **Signal Agreement (Inter-signal Consensus)**:
   Measures consistency across available modalities using weighted mean absolute deviation:
   $$\text{disagreement} = \frac{\sum_{i \in \text{available}} w_i \cdot |s_i - S_{\text{base}}|}{\sum_{i \in \text{available}} w_i}$$
   $$\text{agreement} = \max(0.0, \min(1.0, 1.0 - \text{disagreement}))$$

5. **Engineered Duplicate Confidence**:
   Combines base score, coverage modifier, and consensus agreement:
   $$\text{duplicate\_confidence} = \max(0.0, \min(1.0, S_{\text{base}} \times (0.70 + 0.30 \times \text{coverage}) \times \text{agreement}))$$

6. **Candidate Ranking & Ambiguity Margin**:
   Candidates are scored and ranked descending by `duplicate_confidence`. If multiple candidates exist:
   $$\text{margin} = \text{confidence}_{\text{best}} - \text{confidence}_{\text{runner\_up}}$$
   If $\text{confidence}_{\text{best}} \ge 0.80$ but $\text{margin} < 0.10$ (`VERIFICATION_MIN_MARGIN`), the result is ambiguous between two candidates and returns `NEEDS_REVIEW`.

### Decision States & Thresholds

| Decision | Condition | Meaning |
| :--- | :--- | :--- |
| **`DUPLICATE`** | `duplicate_confidence >= 0.80`, coverage $\ge 0.55$, no conflicts, margin $\ge 0.10$ | Submission matches an existing outlet. |
| **`GENUINE`** | `duplicate_confidence <= 0.35` OR 0 candidates retrieved | Submission represents a distinct new outlet. |
| **`NEEDS_REVIEW`** | Score in $[0.35, 0.80]$, conflicting signals, margin $< 0.10$, or low coverage | Requires manual human review. |

### Deterministic Structured Reason Codes
- `STRONG_MULTIMODAL_MATCH`: High confidence duplicate supported by all three modalities.
- `STRONG_IMAGE_GEO_MATCH`: Strong visual photo match at identical/close GPS location.
- `HIGH_NAME_IMAGE_GEO_MATCH`: High multimodal similarity above duplicate threshold.
- `NO_MATCHING_CANDIDATE`: No candidate outlets retrieved; classified as genuine.
- `CONFLICTING_NAME_IMAGE`: Conflicting evidence (e.g. name similarity $\ge 0.90$ but image similarity $\le 0.50$).
- `AMBIGUOUS_TOP_CANDIDATES`: Top two candidates have nearly identical duplicate confidence (margin $< 0.10$).
- `INSUFFICIENT_EVIDENCE`: Signal coverage below threshold ($< 0.55$) or confidence in uncertain range.
- `POSSIBLE_SAME_BRAND_DIFFERENT_LOCATION`: Identical brand name ($N \ge 0.90$) at distant location ($d \ge 1000\text{ m}$).
- `WEAK_EVIDENCE_REQUIRES_REVIEW`: Missing image with moderate/ambiguous name similarity.
- `LOW_CONFIDENCE_GENUINE`: Duplicate confidence below genuine threshold ($\le 0.35$).

### Configuration Parameters
All parameters are configurable in `.env` and validated at startup:
```env
VERIFICATION_NAME_WEIGHT=0.30
VERIFICATION_IMAGE_WEIGHT=0.45
VERIFICATION_GEO_WEIGHT=0.25
VERIFICATION_GEO_SCALE_METERS=100.0
VERIFICATION_DUPLICATE_THRESHOLD=0.80
VERIFICATION_GENUINE_THRESHOLD=0.35
VERIFICATION_REVIEW_MIN_COVERAGE=0.55
VERIFICATION_STRONG_IMAGE_THRESHOLD=0.90
VERIFICATION_STRONG_IMAGE_GEO_METERS=100.0
VERIFICATION_MIN_MARGIN=0.10
```

### Production Verification API (`POST /verify`)

#### Form Upload (`POST /verify`):
Accepts multipart form fields `name`, `latitude`, `longitude`, and `image` file:
```bash
curl -X POST http://localhost:8000/verify \
  -F "name=Apollo Pharmacy - Indiranagar" \
  -F "latitude=12.9716" \
  -F "longitude=77.5946" \
  -F "image=@/path/to/storefront.jpg"
```

#### JSON Payload (`POST /verify-json`):
```json
{
  "name": "Apollo Pharmacy - Indiranagar",
  "latitude": 12.9716,
  "longitude": 77.5946,
  "image_url": "https://example.com/store.jpg"
}
```

#### Example Verification Response:
```json
{
  "decision": "DUPLICATE",
  "duplicate_confidence": 0.9133,
  "evidence_coverage": 1.0,
  "evidence_agreement": 0.9534,
  "matched_outlet": {
    "outlet_id": "8a02c98d-...",
    "name": "Apollo Pharmacy - Indiranagar 100ft Road",
    "image_url": "https://example.com/apollo_indiranagar.jpg",
    "latitude": 12.9719,
    "longitude": 77.5949
  },
  "evidence": {
    "name_similarity": 0.9142,
    "image_similarity": 0.9854,
    "distance_meters": 46.63,
    "geo_proximity_score": 0.6273
  },
  "matched_methods": ["name", "image", "geo"],
  "reason_codes": ["STRONG_IMAGE_GEO_MATCH", "STRONG_MULTIMODAL_MATCH"],
  "reason_summary": "High duplicate confidence (0.91 >= 0.80). Matched with existing outlet.",
  "candidate_margin": 0.4419,
  "evidence_status": "MATCHED_CANDIDATE",
  "candidates_evaluated": 3
}
```

### Important Disclaimers & Design Principles
1. **Engineered Confidence vs. Calibrated Probability**:
   The `duplicate_confidence` metric is an **engineered evidence fusion score**, **NOT** a calibrated statistical probability. Do not claim "0.91 means 91% probability of duplicate" without empirical calibration against a ground-truth dataset.
2. **Initial Engineering Baseline**:
   Weights ($0.30, 0.45, 0.25$) and thresholds ($0.80, 0.35, 0.55$) are initial engineering values designed to reflect human priors. They will be quantitatively tuned in Phase 6 using a labeled ground-truth dataset.
3. **Deterministic Runtime (No LLMs)**:
   The verification decision engine is 100% deterministic, explainable, and reproducible using Python and PostgreSQL mathematics. No LLMs are invoked at runtime.

### Running Phase 5 Tests & Verification Demo
```powershell
# Run the complete Phase 5 verification test suite (17 comprehensive tests)
python tests/test_verification.py

# Run the live end-to-end verification demonstration against Railway PostgreSQL
python scripts/verify_decision.py
```
