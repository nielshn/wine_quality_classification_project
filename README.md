# Laporan Proyek Machine Learning: Klasifikasi Kualitas Wine

## 1. Domain Proyek

Proyek ini bertujuan untuk mengklasifikasikan kualitas wine (baik red maupun white wine) berdasarkan data sifat kimiawi dari masing-masing sampel. Kualitas wine sangat mempengaruhi persepsi pasar dan penentuan harga, oleh karena itu prediksi kualitas berbasis data sangat krusial untuk efisiensi industri wine.

Masalah ini perlu diselesaikan karena pendekatan manual dalam menilai kualitas wine rawan subjektivitas dan inkonsistensi. Pendekatan machine learning dapat membantu memberikan sistem penilaian yang lebih objektif dan konsisten.

**Referensi:**
- Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). *Modeling wine preferences by data mining from physicochemical properties.* Decision Support Systems, 47(4), 547–553. [Link](https://www.sciencedirect.com/science/article/abs/pii/S0167923609001565)

---

## 2. Business Understanding

### Problem Statement
Bagaimana cara membangun model machine learning yang mampu memprediksi kualitas wine (dalam skala 0-10) berdasarkan data kimiawi dari wine tersebut?

### Goals
- Menghasilkan model klasifikasi kualitas wine yang akurat dan andal.
- Menyediakan pendekatan data-driven untuk mendukung keputusan kualitas produk dalam industri wine.

### Solution Statement
Kami mengusulkan beberapa solusi:
1. **Decision Tree** untuk baseline model.
2. **Random Forest & XGBoost** sebagai model lanjutan untuk meningkatkan performa.
3. **SMOTE** untuk menyeimbangkan distribusi kelas target agar performa model optimal.

Evaluasi dilakukan menggunakan metrik **Accuracy**, **Precision**, **Recall**, **F1-score**, dan **ROC-AUC**.

---

## 3. Data Understanding

- **Jumlah Data:** 4.898 observasi
- **Jumlah Fitur:** 11 fitur + 1 label
- **Link Dataset:** [Wine Quality Dataset - UCI Repository](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)

### Fitur-Fitur:
- `fixed acidity`
- `volatile acidity`
- `citric acid`
- `residual sugar`
- `chlorides`
- `free sulfur dioxide`
- `total sulfur dioxide`
- `density`
- `pH`
- `sulphates`
- `alcohol`
- `quality` (target)

### EDA & Visualisasi:
- Distribusi label menunjukkan ketidakseimbangan kelas (banyak di nilai 5 dan 6).
- Korelasi tinggi ditemukan antara `alcohol` dan `quality`.
- Visualisasi heatmap dan boxplot digunakan untuk insight dan deteksi outlier.

---

## 4. Data Preparation

### Langkah-Langkah:
1. **Pembersihan Data:** Tidak ada missing value yang perlu ditangani.
2. **Normalisasi:** Menggunakan `StandardScaler` untuk menormalkan fitur numerik.
3. **Balancing Data:** Menggunakan teknik **SMOTE** untuk oversampling kelas minoritas.
4. **Feature Engineering:** Membuat fitur turunan seperti `alcohol_density_ratio`.

### Alasan:
- Normalisasi penting karena algoritma seperti SVM dan XGBoost sensitif terhadap skala fitur.
- Balancing diperlukan untuk menghindari bias model ke kelas mayoritas.

---

## 5. Modeling

### Algoritma:
1. **Decision Tree**
2. **Random Forest**
3. **XGBoost**

### Hyperparameter Tuning:
- `Decision Tree:` `max_depth`, `min_samples_split`
- `Random Forest:` `n_estimators`, `max_depth`
- `XGBoost:` `learning_rate`, `n_estimators`, `max_depth`

### Kelebihan & Kekurangan:
- **Decision Tree:** Mudah dipahami, rawan overfitting.
- **Random Forest:** Robust, lebih lama training.
- **XGBoost:** Akurasi tinggi, tapi tuning kompleks.

---

## 6. Evaluation

### Metrik Evaluasi:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **ROC-AUC (jika tersedia)**

### Hasil Evaluasi (Setelah Tuning dan Balancing dengan SMOTE):

| Model          | Accuracy | Precision (class 1) | Recall (class 1) | F1-Score (class 1) |
|----------------|----------|---------------------|------------------|--------------------|
| Decision Tree  | 0.823    | 0.56                | 0.75             | 0.64               |
| SVM            | 0.811    | 0.54                | 0.75             | 0.63               |
| Random Forest  | **0.867**| **0.66**            | **0.76**         | **0.71**           |
| XGBoost        | 0.858    | 0.65                | 0.71             | 0.68               |

> Catatan:
- Kelas `0` (umumnya wine berkualitas biasa) masih mendominasi data, namun performa di kelas `1` (wine berkualitas tinggi) jadi fokus evaluasi karena ini yang lebih kritikal dalam bisnis.
- Model **Random Forest** memberikan keseimbangan terbaik antara recall dan f1-score untuk kelas minoritas (1), yang berarti model ini cukup andal dalam mengidentifikasi wine berkualitas tinggi tanpa terlalu banyak false positive.

### Model Terbaik:
**Random Forest** dipilih sebagai model terbaik karena:
- Mencapai akurasi tertinggi (0.867)
- Memiliki **F1-score tertinggi untuk kelas 1** (0.71)
- Recall untuk kelas minoritas juga paling tinggi (0.76), yang penting untuk meminimalkan false negative terhadap wine berkualitas tinggi.

---



## 7. Kesimpulan

- Model XGBoost terbukti menjadi solusi terbaik dalam klasifikasi kualitas wine.
- Teknik SMOTE efektif dalam menangani ketidakseimbangan kelas.
- Pendekatan machine learning memberikan alternatif yang powerful untuk menilai kualitas produk wine secara objektif dan data-driven.

---

## 8. Struktur Laporan

- Mengikuti urutan: Domain → Business Understanding → Data Understanding → Preparation → Modeling → Evaluation.
- Penjelasan teknis dilengkapi dengan snippet kode (dalam notebook).
- Visualisasi dan resources dapat dimuat dengan baik dalam format markdown.
