# eksperimen-cic-ids

Kode untuk naskah **"Cross-Dataset Generalization of Minority and Low-Rate Attack Detection: An Ablation Study on CIC-IDS2017 and CSE-CIC-IDS2018"**.

## Isi repositori

| File | Fungsi |
|---|---|
| `eksperimen_full_dataset_vscode.ipynb` | Pipeline lengkap: pembacaan data, harmonisasi label dan fitur, deduplikasi, baseline, ablasi A0–A7, skenario S1 (repeated stratified 5-fold × 2) dan S2 (latih CIC-IDS2017, uji CSE-CIC-IDS2018), uji statistik, dan kepentingan fitur. Menghasilkan folder `hasil_ringkas_full`. |
| `analisis_tambahan.py` | Analisis pasca-eksperimen tanpa melatih ulang model: bootstrap 95% CI di S2, macro-F1 tanpa Infiltration, koreksi Holm, corrected resampled t-test (Nadeau & Bengio, 2003), presisi per kelas S2, dan persentase alarm palsu Benign. |
| `requirements.txt` | Daftar paket Python. |

## Data

CIC-IDS2017 (MachineLearningCVE) dan CSE-CIC-IDS2018 tersedia di https://www.unb.ca/cic/datasets/. Data tidak disertakan di repositori ini.

## Cara menjalankan

```bash
pip install -r requirements.txt
# 1. Jalankan notebook dengan RUN_MODE = "FULL" -> menghasilkan folder hasil_ringkas_full
# 2. Jalankan analisis tambahan
python analisis_tambahan.py --hasil hasil_ringkas_full
```

Hasil analisis tambahan disimpan di `hasil_ringkas_full/analisis_tambahan/`. Semua operasi acak memakai seed 42.

## Pengaturan eksperimen (mode FULL)

Seluruh baris dibaca per chunk 250.000; 20% baris Benign CSE-CIC-IDS2018 diambil acak; data latih CIC-IDS2017 dibatasi 20.000 flow per kelas; target SMOTE 2.000; Random Forest 50 pohon; ambang confidence τ = 0,90.
