# Klasifikasi Tingkat Risiko Stroke Berdasarkan Data Klinis dan Gaya Hidup Menggunakan Algoritma TabNet dengan Interpretasi Attention Mechanism

- **Nama**: Muhammad Ridho Ramadhan
- **NIM**: 23.11.5772
- **Dosen**: Anna Baita, S.Kom., M.Kom.

## Deskripsi Proyek

Proyek ini bertujuan membangun model klasifikasi risiko stroke menggunakan arsitektur **TabNet** (Attentive Interpretable Tabular Learning) yang memanfaatkan attention mechanism untuk memberikan interpretabilitas terhadap fitur-fitur klinis dan gaya hidup yang paling berpengaruh terhadap kemunculan stroke.

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
│   ├── raw/              # Data asli (tidak dimodifikasi)
│   ├── interim/          # Data transformasi antara
│   └── processed/        # Data final siap modeling
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_data_preprocessing.ipynb
│   └── 04_dataset_tracking.ipynb
├── reports/
│   └── figures/          # Output grafik EDA
├── metadata/             # Log versioning & hash dataset
├── src/                  # Modul Python reusable
├── requirements.txt
├── dvc.yaml              # Pipeline DVC
├── params.yaml           # Parameter pipeline
└── README.md
```

## Instalasi

```bash
pip install -r requirements.txt
```

## Tahapan Pipeline

1. **Data Collection** → `notebooks/01_data_collection.ipynb`
2. **Exploratory Data Analysis (EDA)** → `notebooks/02_eda.ipynb`
3. **Data Preprocessing** → `notebooks/03_data_preprocessing.ipynb`
4. **Dataset Tracking (DVC)** → `notebooks/04_dataset_tracking.ipynb`

## Tracking Dataset

Proyek ini menggunakan **DVC (Data Version Control)** untuk memastikan reprodusibilitas dataset:

```bash
dvc init
dvc add data/raw/healthcare-dataset-stroke-data.csv
dvc repro
```
