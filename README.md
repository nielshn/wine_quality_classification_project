# 🍇 Wine Quality Classification using Machine Learning

## 1. Domain Project

Wine quality prediction merupakan proyek dalam domain **agrikultur dan pangan**. Proyek ini berfokus pada prediksi kualitas wine menggunakan parameter kimiawi seperti kadar alkohol, pH, dan keasaman. Studi ini merujuk pada riset oleh [Cortez et al., 2009](https://archive.ics.uci.edu/ml/datasets/Wine+Quality) yang mengumpulkan data eksperimen laboratorium terhadap kualitas wine merah dan putih.

> Referensi: Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Modeling wine preferences by data mining from physicochemical properties. Decision Support Systems, 47(4), 547-553. https://archive.ics.uci.edu/ml/datasets/Wine+Quality

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

### Visualisasi

- Countplot distribusi label (Imbalance)
- Heatmap korelasi antar fitur numerik
- Boxplot untuk mendeteksi outlier per fitur

## 4. Data Preparation

### Tahapan

1. **Outlier Removal**: menggunakan metode IQR
2. **Scaling**: menggunakan StandardScaler untuk menormalkan fitur
3. **Balancing**: menerapkan SMOTE untuk menyeimbangkan label
4. **Splitting**: train-test split 80:20 dengan random_state=42

## 5. Modeling

### Model 1: Decision Tree

- Parameter: default (`criterion='gini'`)
- Kelebihan: mudah diinterpretasikan
- Kekurangan: mudah overfitting

### Model 2: Random Forest

- Parameter: default (`n_estimators=100`)
- Kelebihan: stabil, akurat
- Kekurangan: sulit diinterpretasi

### Model 3: XGBoost

- Parameter: `use_label_encoder=False`, `eval_metric='logloss'`
- Kelebihan: efisien dan powerful
- Kekurangan: lebih kompleks

## 6. Evaluation

### Metrik

- Accuracy
- Precision / Recall / F1-score
- ROC AUC

### Hasil Evaluasi

| Model         | Accuracy | Precision | Recall | F1-score | AUC  |
| ------------- | -------- | --------- | ------ | -------- | ---- |
| Decision Tree | 0.82     | 0.57      | 0.64   | 0.61     | 0.76 |
| Random Forest | 0.87     | 0.67      | 0.76   | 0.71     | 0.85 |
| XGBoost       | 0.86     | 0.65      | 0.71   | 0.68     | 0.79 |

- ROC Curve ditampilkan untuk ketiga model
- Confusion Matrix divisualisasikan

## 7. Conclusion

- Random Forest menunjukkan performa terbaik dengan akurasi tinggi dan AUC tertinggi.
- Model XGBoost sebanding namun lebih kompleks.
- Content-based Decision Tree cocok untuk interpretasi awal.
- Random Forest dipilih sebagai model final karena lebih stabil, interpretabel, dan memiliki skor evaluasi tertinggi

## 8. Referensi

[Cortez et al., 2009](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)
[Scikit-learn documentation](https://scikit-learn.org/)
[imbalanced-learn (SMOTE)](https://imbalanced-learn.org/stable/)
[XGBoost documentation](https://xgboost.readthedocs.io/en/latest/)
