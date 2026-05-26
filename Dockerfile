FROM python:3.12-slim

# OS deps minimal untuk pytorch-tabnet (libgomp), build (gcc), dan kebersihan apt cache
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pasang dependency lebih dulu supaya layer cache efektif
COPY requirements.txt .
# Tidak butuh jupyter/notebook di production → exclude utk hemat image size
RUN grep -vE '^(jupyter|ipykernel|notebook|matplotlib|seaborn|plotly|statsmodels)' requirements.txt > requirements-prod.txt \
    && pip install --no-cache-dir -r requirements-prod.txt \
    && pip install --no-cache-dir matplotlib seaborn  # tetap dipakai utk plot fallback bila ada

# Copy kode + artefak model + scaler/encoder yg dibutuhkan untuk inference
COPY src/ ./src/
COPY models/ ./models/
COPY data/processed/scaler.joblib data/processed/encoders.json data/processed/tabnet_meta.json ./data/processed/
COPY .streamlit/ ./.streamlit/

# Default: API. UI di-override di docker-compose.
ENV PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

EXPOSE 9000 8501

# Healthcheck untuk API
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://localhost:${PORT:-9000}/health || exit 1

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "9000"]
