# Klasifikasi Tingkat Risiko Stroke Berdasarkan Data Klinis dan Gaya Hidup Menggunakan Algoritma TabNet dengan Interpretasi Attention Mechanism

- **Nama**: Muhammad Ridho Ramadhan
- **NIM**: 23.11.5772
- **Dosen**: Anna Baita, S.Kom., M.Kom.

## Deskripsi Proyek

Proyek ini membangun sistem **deteksi dini risiko stroke** end-to-end mulai dari pengumpulan data, eksperimen modeling, hingga deployment ke web. Model utama adalah **TabNet** (Attentive Interpretable Tabular Learning) yang memanfaatkan attention mechanism untuk memberikan interpretabilitas terhadap fitur-fitur klinis dan gaya hidup yang paling berpengaruh terhadap kemunculan stroke, dibandingkan dengan tiga baseline klasik (Logistic Regression, Random Forest, XGBoost).

Aspek penting yang ditangani:

- **Imbalanced data** (positif hanya ~4,87%) → SMOTE pada train, class weighting, threshold tuning
- **Primary metric: PR-AUC** (lebih sensitif terhadap kelas minoritas dibanding ROC-AUC)
- **Reproducibility** → DVC pipeline + MLflow tracking + `params.yaml`
- **Modularitas** → kode reusable di [`src/`](src/), bukan hanya notebook
- **Deployment** → REST API (FastAPI) + Web UI Bahasa Indonesia (Streamlit), siap deploy ke VPS

## Dataset

- **Sumber**: Healthcare Dataset Stroke Data (Kaggle - fedesoriano)
- **File**: `data/raw/healthcare-dataset-stroke-data.csv`
- **Ukuran**: 5.110 baris, 12 kolom
- **Target**: `stroke` (0 = tidak stroke, 1 = stroke)

### Fitur

| Kolom               | Deskripsi                                                    | Tipe        |
| ------------------- | ------------------------------------------------------------ | ----------- |
| `id`                | ID pasien (unik)                                             | int         |
| `gender`            | Male / Female / Other                                        | categorical |
| `age`               | Usia pasien                                                  | numeric     |
| `hypertension`      | 0 = tidak, 1 = ya                                            | binary      |
| `heart_disease`     | 0 = tidak, 1 = ya                                            | binary      |
| `ever_married`      | Yes / No                                                     | categorical |
| `work_type`         | Private / Self-employed / Govt_job / children / Never_worked | categorical |
| `Residence_type`    | Urban / Rural                                                | categorical |
| `avg_glucose_level` | Kadar glukosa rata-rata                                      | numeric     |
| `bmi`               | Body Mass Index                                              | numeric     |
| `smoking_status`    | formerly smoked / never smoked / smokes / Unknown            | categorical |
| `stroke`            | Target                                                       | binary      |

## Struktur Proyek

```
Riset Data Mining/
├── data/
│   ├── raw/                             # Dataset asli (di-track DVC)
│   ├── interim/                         # Data transformasi antara
│   └── processed/                       # Data siap modeling (DVC output)
├── notebooks/
│   ├── 01_data_collection.ipynb         # Hash SHA-256 + metadata
│   ├── 02_eda.ipynb                     # 9 figure visualisasi
│   ├── 03_data_preprocessing.ipynb      # Imputasi, encoding, SMOTE, split
│   ├── 04_dataset_tracking.ipynb        # Dokumentasi DVC
│   ├── 05_modeling.ipynb                # Eksperimen modeling (referensi)
│   └── 06_best_model.ipynb              # Leaderboard & visualisasi best model
├── src/
│   ├── __init__.py
│   ├── retrain.py                       # ★ Pipeline retrain tuned (PR-AUC primary)
│   ├── api.py                           # FastAPI backend
│   └── app.py                           # Streamlit UI (Bahasa Indonesia)
├── models/
│   ├── best_model/                      # model.pkl + threshold.joblib
│   ├── best_model.json                  # Metadata best model
│   └── tabnet/                          # TabNet artifact (.zip + network.pt)
├── reports/figures/                     # Plot EDA + evaluasi (DVC output)
├── metadata/                            # Metric JSON (DVC tracked)
├── deploy/
│   └── stroke.conf                      # nginx config untuk VPS
├── docs/
│   ├── DEPLOYMENT_VPS.md                # Panduan deploy VPS
│   ├── Laporan_Proyek_Data_Mining.docx  # Laporan tugas
│   ├── build_word.py                    # Generator laporan
│   └── flowchart_alur_penelitian.png    # Diagram alur
├── .streamlit/config.toml               # Theme light + UI minimal
├── Dockerfile                           # Image untuk API + UI
├── docker-compose.yml                   # Service api + ui
├── dvc.yaml                             # Pipeline 5 stage
├── params.yaml                          # Hyperparameter & config
├── requirements.txt
└── README.md
```

## Instalasi (Lokal / Development)

```bash
python -m venv .venv
source .venv/bin/activate                # macOS/Linux
pip install -r requirements.txt
```

## Pipeline DVC (End-to-End)

Pipeline diorkestrasi via `dvc.yaml` dengan 5 stage berurutan:

```bash
dvc repro             # Jalankan seluruh pipeline (cache-aware)
dvc dag               # Lihat DAG dependency
dvc metrics show      # Lihat semua metrik tracked
```

| # | Stage              | Driver                                | Output utama                                       |
|---|--------------------|---------------------------------------|----------------------------------------------------|
| 1 | `data_collection`  | `notebooks/01_data_collection.ipynb`  | `metadata/01_data_collection.json` (hash + stats)  |
| 2 | `eda`              | `notebooks/02_eda.ipynb`              | 9 figure di `reports/figures/`                     |
| 3 | `preprocess`       | `notebooks/03_data_preprocessing.ipynb` | `data/processed/*` (train/val/test + scaler/encoder) |
| 4 | `modeling`         | `python -m src.retrain`               | `models/best_model/`, `models/tabnet/`, 4 confusion matrix, leaderboard |
| 5 | `evaluate`         | `notebooks/06_best_model.ipynb`       | `reports/figures/model_comparison.png`, dll        |

## Modeling — Tuning + Class Balancing + Threshold Optimization

Modul [`src/retrain.py`](src/retrain.py) melatih 4 algoritma dengan:

- **RandomizedSearchCV** (5-fold stratified, 25 iter) — scoring `average_precision` (PR-AUC)
- **Class balancing** — `scale_pos_weight` untuk XGBoost, `class_weight` untuk LogReg/RF, `weights` untuk TabNet
- **Threshold tuning** — cari threshold yang memaksimalkan F1 di validation set (bukan default 0.5)
- **MLflow logging** — params, metrics, artifact, confusion matrix

### Hasil Leaderboard (sorted by val_pr_auc)

| Model              | Val PR-AUC | Val F1 | Val Recall | Test PR-AUC | Test F1 | Test Recall | Threshold |
|--------------------|------------|--------|------------|-------------|---------|-------------|-----------|
| **Logistic Regression** ★ | **0.2308** | **0.3273** | 0.4865 | **0.1956** | **0.3103** | 0.4865 | 0.606 |
| TabNet             | 0.2092     | 0.3051 | 0.4865     | 0.2093      | 0.2286  | 0.3243      | 0.823     |
| XGBoost            | 0.1573     | 0.2435 | 0.3784     | 0.1245      | 0.2017  | 0.3243      | 0.340     |
| Random Forest      | 0.1568     | 0.2687 | **0.7297** | 0.1288      | 0.2388  | **0.6486**  | 0.250     |

Perbandingan dengan baseline lama (tanpa tuning):
- Test F1 best model naik **+25%** (0.248 → 0.310)
- Test Precision naik **+55%** (0.147 → 0.228)
- Threshold dipindah dari 0.5 default ke 0.606 yang optimal

> **Catatan:** Random Forest memiliki recall tertinggi (0.65) — bisa jadi alternatif jika prioritas adalah meminimalkan kasus stroke yang ter-miss (cocok untuk skenario klinis).

### MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# buka http://127.0.0.1:5000
```

## Load Best Model untuk Inference

```python
import mlflow.sklearn, joblib, numpy as np

model = mlflow.sklearn.load_model("models/best_model")
threshold = joblib.load("models/best_model/threshold.joblib")["threshold"]

proba = model.predict_proba(X)[:, 1]
pred = (proba >= threshold).astype(int)
```

## Deployment

Aplikasi di-deploy sebagai 2 service Docker (API + UI) di belakang nginx native:

```
Internet :443  →  nginx (host)  →  127.0.0.1:8501  (Streamlit UI)
                              ↘   127.0.0.1:9000  (FastAPI)
```

### Jalankan Lokal

Terminal 1 — API:
```bash
uvicorn src.api:app --reload --host 0.0.0.0 --port 9000
# Swagger docs: http://localhost:9000/docs
```

Terminal 2 — UI:
```bash
streamlit run src/app.py
# UI: http://localhost:8501
```

### Jalankan via Docker Compose

```bash
docker compose up -d --build
docker compose ps
docker compose logs -f api ui
```

API expose ke `127.0.0.1:9000`, UI ke `127.0.0.1:8501` (tidak terbuka ke internet — diakses lewat nginx).

### Deploy ke VPS

Lihat panduan lengkap di [`docs/DEPLOYMENT_VPS.md`](docs/DEPLOYMENT_VPS.md).

Ringkasannya:
```bash
# Di VPS Ubuntu
apt install -y nginx certbot python3-certbot-nginx
curl -fsSL https://get.docker.com | sh

# Upload proyek, lalu:
docker compose up -d --build
cp deploy/stroke.conf /etc/nginx/sites-available/
sed -i 's/example.com/domainkamu.com/g' /etc/nginx/sites-available/stroke.conf
ln -s /etc/nginx/sites-available/stroke.conf /etc/nginx/sites-enabled/
certbot --nginx -d stroke.domainkamu.com -d api.stroke.domainkamu.com --redirect
```

## Tech Stack

| Kategori          | Tools                                                    |
|-------------------|----------------------------------------------------------|
| **ML / Modeling** | scikit-learn, XGBoost, pytorch-tabnet, imbalanced-learn |
| **Tracking**      | MLflow (sqlite backend), DVC                             |
| **API**           | FastAPI, uvicorn, Pydantic                               |
| **UI**            | Streamlit (Bahasa Indonesia)                             |
| **Deployment**    | Docker, Docker Compose, nginx, certbot (Let's Encrypt)   |
| **Visualization** | matplotlib, seaborn                                      |

## Lisensi & Sumber

Dataset: [Healthcare Stroke Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) by fedesoriano (Kaggle).

> ⚠ **Disclaimer Medis:** Aplikasi ini bersifat **edukatif dan informatif**, bukan untuk diagnosis medis. Selalu konsultasikan dengan tenaga medis profesional untuk evaluasi kesehatan.
