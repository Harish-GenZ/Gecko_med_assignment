# Stage 1: Build Frontend SPA
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: Python Backend Runtime
FROM python:3.11-slim
WORKDIR /app

# Install system libraries needed for EasyOCR / OpenCV and health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend application code
COPY app/ ./app
COPY alembic/ ./alembic
COPY alembic.ini ./

# Copy built frontend dist from Stage 1
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Expose port and launch uvicorn
EXPOSE 8000
ENV PORT=8000
ENV HOST=0.0.0.0

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
