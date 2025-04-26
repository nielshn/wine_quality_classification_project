# Wine Quality Classification - Final Submission

## 1. Domain Proyek

Proyek ini bertujuan untuk mengklasifikasikan kualitas wine menjadi dua kategori:
- Rendah (Low Quality: label 0)
- Tinggi (High Quality: label 1)

Klasifikasi dilakukan berdasarkan fitur-fitur kimia seperti kadar alkohol, keasaman, pH, kadar gula, dan lain-lain.

**Sumber Data:**
- UCI Machine Learning Repository
- URL: https://archive.ics.uci.edu/ml/datasets/Wine+Quality

Dataset ini menggabungkan dua jenis wine: merah dan putih.

---

## 2. Business Understanding

**Permasalahan:**
Produsen wine ingin memprediksi kualitas produknya secara otomatis, menggunakan data kimiawi hasil produksi, sehingga dapat menghemat waktu dan meningkatkan kontrol kualitas.

**Goal:**
Mengembangkan model machine learning untuk mengklasifikasikan wine berkualitas tinggi dan rendah.

**Solution Statement:**
- Membangun dan membandingkan tiga model: Decision Tree, Random Forest, dan XGBoost.
- Menangani imbalance data menggunakan SMOTE.
- Menggunakan metrik evaluasi seperti Accuracy, Precision, Recall, F1-Score, dan ROC AUC untuk memilih model terbaik.

---

## 3. Data Understanding

**Informasi Dataset:**
- **Jumlah Baris:** 6497
- **Jumlah Kolom:** 13

**Kondisi Data:**
- Tidak ada missing value.
- Terdapat beberapa duplikasi data.
- Ditemukan beberapa outlier pada fitur numerik.

**Fitur-fitur:**
1. fixed acidity
2. volatile acidity
3. citric acid
4. residual sugar
5. chlorides
6. free sulfur dioxide
7. total sulfur dioxide
8. density
9. pH
10. sulphates
11. alcohol
12. type (red/white)
13. quality (nilai 0-10)
14. quality_label (hasil transformasi klasifikasi 0 atau 1)

---

## 4. Data Preparation

Pada tahap ini dilakukan beberapa tahapan pemrosesan data:

- **Handling Outlier:**
  - Menghapus data ekstrem menggunakan metode IQR untuk menjaga stabilitas model.

- **Feature Engineering:**
  - Membuat kolom baru `quality_label` berdasarkan skor `quality`.
  - Label 1 untuk skor >=7 (tinggi), Label 0 untuk skor <7 (rendah).

- **Splitting Data:**
  - Data dibagi menjadi training dan testing set dengan rasio 80:20.

- **Feature Scaling:**
  - Melakukan normalisasi fitur numerik menggunakan `StandardScaler` agar fitur memiliki skala yang seragam.

- **Balancing Data:**
  - Menggunakan `SMOTE` (Synthetic Minority Oversampling Technique) untuk mengatasi masalah imbalance label.

---

## 5. Modeling

### Model 1: Decision Tree Classifier

**Cara Kerja:**
- Membentuk pohon keputusan berdasarkan fitur yang paling baik memisahkan kelas.
- Pemilihan fitur didasarkan pada pengukuran impuritas seperti Gini Impurity.

**Parameter:**
- `criterion='gini'` (default)
- `random_state=42`

**Kelebihan:**
- Interpretasi mudah, dapat divisualisasikan.

**Kekurangan:**
- Rentan overfitting tanpa teknik pruning.

---

### Model 2: Random Forest Classifier

**Cara Kerja:**
- Membentuk banyak pohon keputusan di atas subset data berbeda, lalu mengambil voting mayoritas untuk prediksi.
- Mengurangi variansi model tunggal.

**Parameter:**
- `n_estimators=100` (default)
- `random_state=42`

**Kelebihan:**
- Lebih akurat dan tahan terhadap overfitting.

**Kekurangan:**
- Sulit untuk interpretasi model secara keseluruhan.

---

### Model 3: XGBoost Classifier

**Cara Kerja:**
- Menggunakan teknik boosting berbasis gradient descent untuk mengurangi error secara iteratif.
- Setiap model berikutnya fokus memperbaiki kesalahan dari model sebelumnya.

**Parameter:**
- `use_label_encoder=False`
- `eval_metric='logloss'`

**Kelebihan:**
- Sangat kuat dan efektif untuk dataset tabular besar.

**Kekurangan:**
- Membutuhkan tuning parameter yang lebih kompleks.

---

## 6. Evaluation

Model dievaluasi menggunakan metrik berikut:

- **Accuracy:** Persentase prediksi yang benar.
- **Precision, Recall, F1-Score:** Untuk menilai keseimbangan performa model.
- **ROC AUC Curve:** Untuk menilai kemampuan model membedakan antar kelas.

**Hasil evaluasi:**
- Random Forest dan XGBoost menunjukkan kinerja lebih baik dibandingkan Decision Tree setelah balancing dan tuning sederhana.

Visualisasi ROC Curve membuktikan AUC Score tertinggi didapatkan oleh Random Forest.

---

## 7. Conclusion

Model Random Forest dipilih sebagai model terbaik untuk mengklasifikasikan kualitas wine karena memberikan skor akurasi, recall, precision, dan AUC terbaik setelah balancing data menggunakan SMOTE.

Untuk meningkatkan akurasi lebih lanjut, model dapat dioptimasi dengan teknik hyperparameter tuning lanjutan seperti Randomized Search atau Grid Search pada n_estimators, max_depth, dan learning_rate.

