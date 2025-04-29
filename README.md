# Wine Quality Classification - Final Submission

## 1. Domain Proyek

Wine adalah salah satu produk konsumsi bernilai tinggi yang produksinya sangat dipengaruhi oleh faktor kualitas. Penentuan kualitas wine secara manual menggunakan pengecekan sensorik membutuhkan biaya dan waktu yang tinggi. Oleh karena itu, diperlukan pendekatan otomatisasi berbasis data untuk mengklasifikasikan kualitas wine berdasarkan parameter kimiawi.

### Latar Belakang Masalah
Masalah dalam penilaian kualitas wine adalah tidak efisiennya metode tradisional yang mengandalkan pengecekan manusia dan uji rasa. Dengan meningkatnya produksi wine secara global, kebutuhan akan sistem penilaian otomatis semakin besar.

### Mengapa Masalah Ini Penting?
- **Skalabilitas**: metode otomatis bisa digunakan untuk ribuan sampel per hari.
- **Konsistensi**: mengurangi subjektivitas penilaian manusia.
- **Efisiensi**: mempercepat proses quality control dalam produksi.

### Referensi Terkait
- Cortez, Paulo, et al. "Modeling wine preferences by data mining from physicochemical properties." Decision Support Systems 47.4 (2009): 547-553.

---

## 2. Business Understanding

### Problem Statement
Bagaimana memprediksi kualitas wine secara otomatis (tinggi/rendah) berdasarkan parameter kimiawi untuk meningkatkan efisiensi proses produksi dan menjaga konsistensi kualitas produk?

### Goals
- Membangun model klasifikasi untuk memprediksi label kualitas wine (0 atau 1).
- Meningkatkan akurasi model hingga >85%.
- Memilih model terbaik dari beberapa kandidat dengan metrik evaluasi.

### Solution Statement
- Menggunakan 3 algoritma: Decision Tree, Random Forest, dan XGBoost.
- Melakukan balancing data menggunakan SMOTE.
- Evaluasi performa dengan metrik: Accuracy, Precision, Recall, F1-Score, ROC AUC.

---

## 3. Data Understanding

Dataset berasal dari UCI Machine Learning Repository:
https://archive.ics.uci.edu/ml/datasets/Wine+Quality

### Informasi Dataset
- Jumlah baris: 6497
- Jumlah kolom: 13 + 1 kolom tambahan (quality_label)

### Kondisi Data:
- Tidak ada missing value
- Terdapat data duplikat
- Ditemukan outlier pada beberapa fitur numerik (ditangani dengan metode IQR)

### Fitur:
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
13. quality (0–10)
14. quality_label (binary label target)

Visualisasi EDA juga dilakukan: korelasi antar fitur, distribusi label kualitas, boxplot outlier.

---

## 4. Data Preparation

- **Outlier Handling**: menggunakan metode IQR untuk membersihkan data ekstrem
- **Label Encoding**: label klasifikasi dibuat berdasarkan skor kualitas (>=7: 1, <7: 0)
- **Splitting Data**: 80% untuk train, 20% untuk test
- **Feature Scaling**: menggunakan StandardScaler agar semua fitur memiliki skala seragam
- **Balancing Data**: menggunakan SMOTE untuk menyamakan distribusi kelas minoritas

---

## 5. Modeling

### Model 1: Decision Tree Classifier
- **Cara kerja**: Membangun pohon keputusan berdasarkan impuritas (Gini)
- **Parameter**: `criterion='gini'`, `random_state=42`
- **Kelebihan**: Mudah diinterpretasi
- **Kekurangan**: Rentan overfitting

### Model 2: Random Forest Classifier
- **Cara kerja**: Ensemble dari banyak pohon, prediksi berdasarkan voting mayoritas
- **Parameter**: `n_estimators=100`, `random_state=42`
- **Kelebihan**: Akurat dan tahan overfitting
- **Kekurangan**: Kurang interpretatif

### Model 3: XGBoost Classifier
- **Cara kerja**: Boosting berbasis gradient descent, iteratif memperbaiki error
- **Parameter**: `use_label_encoder=False`, `eval_metric='logloss'`
- **Kelebihan**: Performa tinggi
- **Kekurangan**: Butuh tuning lanjutan

---

## 6. Evaluation

### Metrik Evaluasi yang Digunakan
- Accuracy: rasio prediksi benar
- Precision, Recall, F1-Score: untuk menangani data imbalance
- ROC AUC: menilai kemampuan model membedakan antar kelas

### Hasil Evaluasi Model
| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| Decision Tree | 84.0% | 0.59 | 0.58 | 0.58 | 0.74 |
| Random Forest | **89.3%** | 0.81 | 0.59 | 0.68 | **0.86** |
| XGBoost | 88.9% | 0.76 | 0.63 | 0.69 | 0.84 |

### Evaluasi terhadap Business Understanding
- Problem statement telah terjawab dengan membangun 3 model untuk klasifikasi kualitas wine.
- Tujuan untuk mencapai akurasi >85% berhasil dicapai oleh Random Forest dan XGBoost.
- Random Forest dipilih karena memberikan metrik terbaik secara keseluruhan dan stabil.
- Solusi ini bisa langsung diterapkan untuk membantu tim QC di industri wine.

---

## 7. Conclusion

Model Random Forest dipilih sebagai model terbaik karena memberikan hasil prediksi paling akurat dan konsisten. Dengan akurasi 89%, AUC 0.86, serta dukungan teknik balancing dan scaling, sistem ini siap digunakan untuk mendukung proses quality control industri wine secara otomatis.

Model ini berhasil menjawab tantangan yang ada, dan dapat dikembangkan lebih lanjut melalui hyperparameter tuning lanjutan atau deployment ke dalam sistem produksi.

---

**Disusun oleh:** Daniel Siahaan - Laskar Ai 2025
