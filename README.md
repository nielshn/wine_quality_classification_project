# 🍇 Wine Quality Classification using Machine Learning

## 1. Domain Project

Wine quality prediction merupakan proyek dalam domain **agrikultur dan pangan**. Proyek ini berfokus pada prediksi kualitas wine menggunakan parameter kimiawi seperti kadar alkohol, pH, dan keasaman. Studi ini merujuk pada riset oleh [Cortez et al., 2009](https://archive.ics.uci.edu/ml/datasets/Wine+Quality) yang mengumpulkan data eksperimen laboratorium terhadap kualitas wine merah dan putih.

> Referensi: Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Modeling wine preferences by data mining from physicochemical properties. Decision Support Systems, 47(4), 547-553.

## 2. Business Understanding

### Problem Statement

Produsen wine menghadapi tantangan dalam menilai kualitas produk mereka secara efisien. Pengujian organoleptik membutuhkan waktu dan biaya besar, sementara parameter kimiawi tersedia saat produksi.

### Goals

- Mengembangkan model klasifikasi kualitas wine (Low: 3-6, High: 7-9).
- Menyajikan model yang mampu menangani ketidakseimbangan data dan memberikan interpretasi yang baik.

### Solution Statement

1. **Decision Tree** sebagai model baseline yang mudah diinterpretasi.
2. **Random Forest** sebagai ensemble untuk meningkatkan stabilitas dan akurasi.
3. **XGBoost** untuk peningkatan performa dan penanganan data tabular.
4. **SMOTE** untuk mengatasi kelas minoritas (High quality).

## 3. Data Understanding

### Dataset

- Sumber: [UCI Machine Learning Repository - Wine Quality](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)
- Jumlah data:

  - Red wine: 1599 sampel
  - White wine: 4898 sampel
  - Total: 6497 sampel

### Kondisi Data

- Tidak ada nilai hilang
- Terdapat beberapa duplikat dan outlier pada fitur `residual sugar`, `chlorides`, dan `density`
- Target `quality` dibagi menjadi label biner: 0 = Low (3-6), 1 = High (7-9)

### Penjelasan Fitur

| Fitur                | Deskripsi                                             | Satuan |
| -------------------- | ----------------------------------------------------- | ------ |
| fixed acidity        | Kandungan asam tartarat utama                         | g/dm³  |
| volatile acidity     | Asam asetat, terlalu tinggi berpengaruh buruk ke rasa | g/dm³  |
| citric acid          | Asam sitrat, pengawet alami                           | g/dm³  |
| residual sugar       | Gula yang tersisa setelah fermentasi                  | g/dm³  |
| chlorides            | Kandungan garam (NaCl)                                | g/dm³  |
| free sulfur dioxide  | SO2 bebas, mencegah pertumbuhan mikroba               | ppm    |
| total sulfur dioxide | Total SO2 (bebas + terikat)                           | ppm    |
| density              | Massa jenis wine                                      | g/cm³  |
| pH                   | Tingkat keasaman                                      | -      |
| sulphates            | Pengawet tambahan                                     | g/dm³  |
| alcohol              | Kandungan alkohol                                     | % vol  |

### Insight dari Visualisasi

- Label `quality` sangat tidak seimbang, mayoritas pada kelas 5 dan 6.
- Korelasi tinggi ditemukan antara `density` dan `residual sugar` (positif), serta antara `alcohol` dan `quality` (positif moderat).
- Outlier mencolok pada `chlorides` dan `residual sugar`, yang dapat memengaruhi performa model.

## 4. Data Preparation

### Tahapan

1. **Train-Test Split**: Data dibagi terlebih dahulu (80:20) sebelum proses lainnya untuk menghindari _data leakage_ saat proses balancing.
2. **Outlier Removal**: Menggunakan metode IQR untuk membersihkan nilai ekstrim yang dapat memengaruhi distribusi model.
3. **Scaling**: Menggunakan StandardScaler untuk menormalkan skala fitur, penting agar model berbasis jarak bekerja optimal.
4. **Balancing**: SMOTE digunakan untuk membuat distribusi label seimbang. Dipilih karena metode ini mensintesis data baru dari minoritas (kelas 1), bukan hanya duplikasi.

## 5. Modeling

### Model 1: Decision Tree

- Parameter: default (`criterion='gini'`)
- Cara kerja: Membagi data secara rekursif berdasarkan fitur yang menghasilkan _impurity_ terkecil pada setiap node.
- Kelebihan: Sangat interpretatif, hasilnya bisa divisualisasikan.
- Kekurangan: Rentan overfitting jika tidak dipangkas atau diatur depth-nya.

### Model 2: Random Forest

- Parameter: default (`n_estimators=100`)
- Cara kerja: Kombinasi banyak pohon keputusan (bagging) untuk mengurangi variansi dan overfitting. Hasil akhir dari voting mayoritas.
- Kelebihan: Stabil, tahan terhadap overfitting, mampu menangani data tidak linear.
- Kekurangan: Interpretasi sulit karena kompleksitas banyak pohon.

### Model 3: XGBoost

- Parameter: `use_label_encoder=False`, `eval_metric='logloss'`
- Cara kerja: Ensemble berbasis boosting, setiap pohon baru dibangun untuk mengoreksi kesalahan dari pohon sebelumnya. Gunakan gradient descent untuk optimisasi.
- Kelebihan: Cepat, efisien, menangani missing value dan regularisasi.
- Kekurangan: Kompleksitas tinggi dan butuh tuning parameter.

## 6. Evaluation

### Metrik Evaluasi

- **Accuracy**: Proporsi prediksi benar dari total data.
- **Precision**: Ketepatan model dalam memprediksi kelas High.
- **Recall**: Seberapa banyak kelas High yang berhasil dikenali.
- **F1-score**: Harmonik antara precision dan recall.
- **AUC (ROC)**: Luas di bawah kurva ROC; mengukur kemampuan model membedakan kelas secara keseluruhan.

### Hasil Evaluasi (sesuai output notebook)

| Model         | Accuracy | Precision | Recall | F1-score | AUC  |
| ------------- | -------- | --------- | ------ | -------- | ---- |
| Decision Tree | 0.82     | 0.58      | 0.75   | 0.65     | 0.76 |
| Random Forest | 0.87     | 0.68      | 0.78   | 0.72     | 0.85 |
| XGBoost       | 0.86     | 0.66      | 0.74   | 0.70     | 0.83 |

### Interpretasi

- **ROC Curve** menunjukkan Random Forest memiliki kurva paling mendekati titik kiri atas → kemampuan klasifikasi tinggi.
- **Confusion Matrix**: Random Forest memiliki false negative paling rendah dibanding dua model lainnya → bagus untuk meminimalkan wine berkualitas tinggi yang terlewat. Confusion Matrix menunjukkan Random Forest mampu mendeteksi lebih banyak kelas ‘High’ dibanding model lain, yang penting untuk mengurangi missed detection pada wine berkualitas tinggi.
- **F1-score dan AUC** menjadi dasar utama pemilihan model akhir karena fokus pada keseimbangan ketepatan dan cakupan, serta kemampuan diskriminasi antar kelas.

## 7. Conclusion

- Random Forest menunjukkan performa terbaik berdasarkan **AUC (0.85)** dan **F1-score (0.72)**, menunjukkan keseimbangan optimal antara presisi dan recall.
- XGBoost menyusul dekat namun lebih kompleks dan tidak jauh berbeda dari segi metrik.
- Decision Tree tetap berguna untuk baseline dan interpretasi awal.
- Random Forest dipilih sebagai **model final** karena mampu menangani ketidakseimbangan data dengan baik, memberikan skor evaluasi tinggi, dan cukup stabil untuk dipakai pada data nyata.

## 8. Referensi

- [Cortez et al., 2009](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)
- [Scikit-learn documentation](https://scikit-learn.org/)
- [imbalanced-learn (SMOTE)](https://imbalanced-learn.org/stable/)
- [XGBoost documentation](https://xgboost.readthedocs.io/en/latest/)
