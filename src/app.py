"""
Streamlit UI untuk deteksi dini risiko stroke (per pasien).

Run:
    streamlit run src/app.py
"""
from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("STROKE_API_URL", "http://localhost:9000")

st.set_page_config(
    page_title="Deteksi Dini Risiko Stroke",
    page_icon="🧠",
    layout="wide",
)

# ---------------------------------------------------------------- terjemahan label kategorikal
LABELS = {
    "gender": {"Female": "Perempuan", "Male": "Laki-laki"},
    "ever_married": {"No": "Belum / Tidak Menikah", "Yes": "Sudah Menikah"},
    "work_type": {
        "Govt_job": "Pegawai Pemerintah / PNS",
        "Never_worked": "Belum Pernah Bekerja",
        "Private": "Karyawan Swasta",
        "Self-employed": "Wiraswasta",
        "children": "Anak-anak (belum bekerja)",
    },
    "Residence_type": {"Rural": "Pedesaan", "Urban": "Perkotaan"},
    "smoking_status": {
        "Unknown": "Tidak Diketahui",
        "formerly smoked": "Pernah Merokok (sudah berhenti)",
        "never smoked": "Tidak Pernah Merokok",
        "smokes": "Masih Merokok Aktif",
    },
}

RISK_LABEL_ID = {
    "Low": "Rendah",
    "Moderate": "Sedang",
    "High": "Tinggi",
    "Very High": "Sangat Tinggi",
}


def label_of(field: str, raw_value: str) -> str:
    """Ambil label Bahasa Indonesia dari nilai mentah; fallback ke nilai aslinya."""
    return LABELS.get(field, {}).get(raw_value, raw_value)


# ---------------------------------------------------------------- header
st.title("Aplikasi Deteksi Dini Risiko Stroke")
st.caption(
    "Aplikasi ini membantu **mendeteksi dini kemungkinan terjadinya stroke** "
    "berdasarkan data klinis dan gaya hidup pasien. "
    "Hasil prediksi bersifat **informatif** dan **bukan diagnosis medis** — "
    "silakan konsultasikan ke dokter untuk evaluasi lanjutan."
)

# ---------------------------------------------------------------- API health
try:
    schema = requests.get(f"{API_URL}/schema", timeout=3).json()
except Exception as e:
    st.error(
        f"❌ Layanan prediksi tidak dapat dihubungi di {API_URL}.\n\n"
        f"Pastikan API sudah dijalankan dengan perintah: `uvicorn src.api:app --reload`\n\n"
        f"Detail kesalahan: {e}"
    )
    st.stop()

# ---------------------------------------------------------------- input form
st.subheader("Isi Data Pasien")
st.caption("Lengkapi seluruh informasi di bawah ini, lalu tekan tombol **Prediksi Risiko**.")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Data Diri**")
    gender = st.selectbox(
        "Jenis Kelamin",
        schema["categorical"]["gender"],
        format_func=lambda v: label_of("gender", v),
        help="Pilih sesuai dengan jenis kelamin pasien",
    )
    age = st.slider("Usia (tahun)", 0, 120, 45,
                    help="Geser untuk memilih usia pasien")
    ever_married = st.selectbox(
        "Status Pernikahan",
        schema["categorical"]["ever_married"],
        format_func=lambda v: label_of("ever_married", v),
    )
    residence = st.selectbox(
        "Tempat Tinggal",
        schema["categorical"]["Residence_type"],
        format_func=lambda v: label_of("Residence_type", v),
    )

with col2:
    st.markdown("**Kondisi Kesehatan**")
    hypertension = st.radio(
        "Tekanan Darah Tinggi (Hipertensi)",
        [0, 1],
        format_func=lambda x: "Tidak" if x == 0 else "Ya, pernah/sedang hipertensi",
        horizontal=True,
        help="Pilih 'Ya' jika pasien pernah didiagnosis hipertensi",
    )
    heart_disease = st.radio(
        "Riwayat Penyakit Jantung",
        [0, 1],
        format_func=lambda x: "Tidak" if x == 0 else "Ya, ada riwayat",
        horizontal=True,
        help="Pilih 'Ya' jika ada riwayat penyakit jantung",
    )
    avg_glucose = st.number_input(
        "Rata-rata Kadar Gula Darah (mg/dL)",
        min_value=0.0, max_value=400.0, value=100.0, step=0.1,
        help="Normal: 70–140 mg/dL. Diabetes: di atas 200 mg/dL",
    )
    bmi = st.number_input(
        "Indeks Massa Tubuh / BMI",
        min_value=10.0, max_value=80.0, value=25.0, step=0.1,
        help="BMI = berat badan (kg) ÷ tinggi badan² (m). Normal 18.5-24.9",
    )

with col3:
    st.markdown("**Pekerjaan & Gaya Hidup**")
    work_type = st.selectbox(
        "Jenis Pekerjaan",
        schema["categorical"]["work_type"],
        format_func=lambda v: label_of("work_type", v),
    )
    smoking = st.selectbox(
        "Status Merokok",
        schema["categorical"]["smoking_status"],
        format_func=lambda v: label_of("smoking_status", v),
    )

st.divider()
if st.button("Prediksi Risiko", type="primary", use_container_width=True):
    payload = {
        "gender": gender, "age": age, "hypertension": hypertension,
        "heart_disease": heart_disease, "ever_married": ever_married,
        "work_type": work_type, "Residence_type": residence,
        "avg_glucose_level": avg_glucose, "bmi": bmi, "smoking_status": smoking,
    }
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        r.raise_for_status()
        res = r.json()
    except Exception as e:
        st.error(f"Gagal mengirim data ke layanan prediksi: {e}")
        st.stop()

    # Visual result
    st.subheader("Hasil Prediksi")
    c1, c2, c3 = st.columns(3)
    c1.metric("Kemungkinan Stroke", f"{res['probability']*100:.2f}%")
    c2.metric("Batas Keputusan", f"{res['threshold']*100:.2f}%")
    c3.metric(
        "Kesimpulan",
        "⚠ Berisiko Stroke" if res["prediction"] else "✓ Tidak Berisiko Stroke",
    )

    # Risk bar
    prob = res["probability"]
    risk_id = RISK_LABEL_ID.get(res["risk_label"], res["risk_label"])
    color = {"Low": "green", "Moderate": "orange",
             "High": "red", "Very High": "red"}[res["risk_label"]]
    st.markdown(f"### Tingkat Risiko: :{color}[**{risk_id}**]")
    st.progress(min(prob, 1.0))

    if res["prediction"] == 1:
        st.warning(
            "⚠ **Perhatian:** Model mengindikasikan adanya **kemungkinan risiko stroke yang tinggi** "
            "pada pasien ini. Sangat disarankan untuk segera **berkonsultasi dengan dokter** "
            "guna pemeriksaan medis lebih lanjut.\n\n"
            "_Catatan: hasil ini bersifat informatif, bukan diagnosis medis._"
        )
    else:
        st.success(
            f"✓ Kemungkinan stroke ({prob*100:.2f}%) berada **di bawah batas keputusan** "
            f"({res['threshold']*100:.2f}%). Pertahankan pola hidup sehat: olahraga teratur, "
            f"makanan bergizi, tidur cukup, dan kontrol kesehatan rutin."
        )

    # Ringkasan input agar mudah dibaca ulang
    with st.expander("📋 Ringkasan Data yang Dimasukkan"):
        ringkasan = pd.DataFrame({
            "Atribut": [
                "Jenis Kelamin", "Usia", "Tekanan Darah Tinggi", "Penyakit Jantung",
                "Status Pernikahan", "Pekerjaan", "Tempat Tinggal",
                "Kadar Gula Darah", "BMI", "Status Merokok",
            ],
            "Nilai": [
                label_of("gender", gender),
                f"{age} tahun",
                "Ya" if hypertension else "Tidak",
                "Ya" if heart_disease else "Tidak",
                label_of("ever_married", ever_married),
                label_of("work_type", work_type),
                label_of("Residence_type", residence),
                f"{avg_glucose:.1f} mg/dL",
                f"{bmi:.1f}",
                label_of("smoking_status", smoking),
            ],
        })
        st.dataframe(ringkasan, hide_index=True, use_container_width=True)

    with st.expander("Detail Teknis (untuk pengembang)"):
        st.json(res)
