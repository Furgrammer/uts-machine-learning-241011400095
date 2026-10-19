# Tugas UTS Machine Learning

- **Nama:** Muhammad Al Furqon
- **NIM:** 241011400095
- **Kelas:** 05TPLM001

## Ringkasan Tugas

Tugas ini menerapkan **Linear Regression** untuk memperkirakan penjualan berdasarkan biaya iklan dan **Logistic Regression** untuk mengklasifikasikan email spam. Model dan metrik utama diimplementasikan menggunakan NumPy; scikit-learn digunakan hanya untuk bagian bonus.

- `random_state`: **95** (dua digit terakhir NIM)
- Dataset iklan: **40 baris**, dibagi menjadi 32 data training dan 8 data testing.
- Dataset email: **60 baris** (35 bukan spam dan 25 spam), dibagi menjadi 45 data training dan 15 data testing.

## Bagian A — Linear Regression

Model dilatih dengan Normal Equation. Persamaan regresinya:

```text
Penjualan = 27.8357 + 9.1980 × BiayaIklan
```

| Data | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| Training | 4.5004 | 38.1144 | 6.1737 | 0.9422 |
| Testing | 4.3484 | 27.9564 | 5.2874 | 0.9328 |

Pada data testing, model menjelaskan sekitar **93,28%** variasi penjualan. Prediksi penjualan untuk biaya iklan Rp4,5 juta, Rp7 juta, dan Rp12 juta masing-masing sekitar **69,2266**, **92,2216**, dan **138,2116** juta rupiah. Prediksi Rp12 juta merupakan ekstrapolasi karena berada di luar rentang data iklan yang diamati.

### Eksperimen Gradient Descent

- Learning rate **0,001** masih bergerak menuju solusi dan belum mencapai parameter Normal Equation setelah 5.000 iterasi.
- Learning rate **0,02** mencapai hasil yang praktis sama dengan Normal Equation.
- Learning rate **0,1** divergen pada iterasi ke-191 karena langkah pembaruan terlalu besar.

## Bagian B — Logistic Regression

Model Gradient Descent menghasilkan persamaan probabilitas:

```text
P(Spam) = sigmoid(-15.006311 + 1.368714 × JumlahLink + 1.118385 × KataPromo)
```

Log loss training turun dari **0.693147** menjadi **0.134641** setelah 10.000 iterasi.

Dengan threshold **0,5**, hasil evaluasi pada 15 data testing adalah:

| Metrik | Hasil |
|---|---:|
| Confusion matrix (baris aktual: bukan spam, spam; kolom prediksi: bukan spam, spam) | `[[8, 0], [3, 4]]` |
| Accuracy | 0.8000 |
| Precision | 1.0000 |
| Recall | 0.5714 |
| F1-score | 0.7273 |

Perbandingan threshold:

| Threshold | TN | FP | FN | TP | Precision | Recall |
|---:|---:|---:|---:|---:|---:|---:|
| 0.3 | 8 | 0 | 3 | 4 | 1.0000 | 0.5714 |
| 0.5 | 8 | 0 | 3 | 4 | 1.0000 | 0.5714 |
| 0.7 | 8 | 0 | 4 | 3 | 1.0000 | 0.4286 |

Pada empat email contoh, model memprediksi E1 dan E2 sebagai bukan spam serta E3 dan E4 sebagai spam (menggunakan threshold 0,3).

## Bonus — Perbandingan dengan scikit-learn

- **Linear Regression:** NumPy dan scikit-learn menghasilkan metrik testing yang sama, termasuk **R² = 0.9328**. Selisih maksimum prediksi keduanya hanya sekitar `1.49 × 10⁻¹³`.
- **Logistic Regression:** dengan threshold 0,3, implementasi NumPy, scikit-learn dengan regularisasi L2 default, dan scikit-learn tanpa regularisasi sama-sama memperoleh **accuracy 0.8000**, **precision 1.0000**, **recall 0.5714**, dan **F1-score 0.7273** pada data testing. Log loss masing-masing adalah **0.851136**, **0.910028**, dan **6.907755**.

## File

- [Notebook tugas](./tugas_UTS.ipynb)
- [Dataset iklan](./Ads_Data.csv)
- [Dataset email](./Email_Data.csv)
