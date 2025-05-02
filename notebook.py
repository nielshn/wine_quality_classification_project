# Wine Quality Classification Final Submission

# === 1. Domain Proyek ===
"""
Proyek ini bertujuan untuk mengklasifikasikan kualitas wine menjadi dua kelas:
- Rendah (Low Quality, 0)
- Tinggi (High Quality, 1)
berdasarkan fitur-fitur kimiawi seperti keasaman, pH, kadar alkohol, dan lain-lain.

Sumber Data: UCI Machine Learning Repository
URL: https://archive.ics.uci.edu/ml/datasets/Wine+Quality
"""

# === 2. Business Understanding ===
"""
Permasalahan:
Produsen wine ingin mengoptimalkan proses produksi dengan memprediksi kualitas produk berdasarkan parameter kimia selama proses produksi.

Goal:
Mengembangkan model klasifikasi yang dapat mengkategorikan kualitas wine sebagai 'Rendah' atau 'Tinggi'.

Solution Statement:
- Membandingkan beberapa algoritma ML: Decision Tree, Random Forest, dan XGBoost.
- Menggunakan balancing data dengan SMOTE.
- Menggunakan evaluasi berbasis Accuracy, Precision, Recall, F1-Score, dan AUC.
"""

# === 3. Data Understanding ===
# 1. Import Libraries
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load dataset
red = pd.read_csv("winequality-red.csv", sep=';')
white = pd.read_csv("winequality-white.csv", sep=';')
red['type'], white['type'] = 'red', 'white'
wine = pd.concat([red, white], axis=0)
wine['quality_label'] = wine['quality'].apply(lambda x: 1 if x >= 7 else 0)

# 3. EDA
# 3. Exploratory Data Analysis (EDA)

# Distribusi label
plt.figure(figsize=(6, 4))
sns.countplot(x='quality_label', data=wine)
plt.title("Distribusi Kualitas Wine")
plt.xlabel("Label Kualitas (0 = Rendah, 1 = Tinggi)")
plt.ylabel("Jumlah")
plt.tight_layout()
plt.show()

# Korelasi antar fitur (filter kolom numerik saja)
plt.figure(figsize=(12, 10))
numerical_cols = wine.select_dtypes(include=['float64', 'int64'])
sns.heatmap(numerical_cols.corr(), annot=True, cmap='coolwarm')
plt.title("Heatmap Korelasi Antar Fitur Numerik")
plt.tight_layout()
plt.show()

# Boxplot per fitur numerik
features = wine.select_dtypes(include=['float64', 'int64']).drop(
    columns=['quality', 'quality_label']).columns
plt.figure(figsize=(20, 15))
for i, col in enumerate(features):
    plt.subplot(4, 3, i + 1)
    sns.boxplot(x='quality_label', y=col, data=wine)
    plt.title(f"Boxplot: {col} vs Quality Label")
plt.tight_layout()
plt.show()
# Uraian Fitur:
"""
- fixed acidity: tingkat keasaman tetap
- volatile acidity: tingkat keasaman volatil
- citric acid: konsentrasi asam sitrat
- residual sugar: sisa gula
- chlorides: kadar klorida
- free sulfur dioxide: sulfur dioksida bebas
- total sulfur dioxide: total sulfur dioksida
- density: kerapatan
- pH: tingkat keasaman
- sulphates: konsentrasi sulfat
- alcohol: kadar alkohol
- type: jenis wine (merah/putih)
- quality: nilai kualitas asli (0-10)
- quality_label: label klasifikasi (0 rendah, 1 tinggi)
"""

# Korelasi antar fitur numerik
plt.figure(figsize=(12, 10))
sns.heatmap(wine.drop(['quality', 'type'], axis=1).corr(),
            annot=True, cmap='coolwarm')
plt.title('Korelasi antar Fitur')
plt.tight_layout()
plt.show()

# === 4. Data Preparation ===
numerical_cols = wine.select_dtypes(include=["float64", "int64"]).drop(
    columns=["quality", "quality_label"]).columns


def remove_outliers(df, columns):
    cleaned_df = df.copy()
    for col in columns:
        Q1 = cleaned_df[col].quantile(0.25)
        Q3 = cleaned_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        cleaned_df = cleaned_df[(cleaned_df[col] >= lower)
                                & (cleaned_df[col] <= upper)]
    return cleaned_df


wine_clean = remove_outliers(wine.copy(), numerical_cols)
X = wine_clean.drop(['quality', 'quality_label', 'type'], axis=1)
y = wine_clean['quality_label']
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

sm = SMOTE(random_state=42)
X_train_smote, y_train_smote = sm.fit_resample(X_train_scaled, y_train)

# === 5. Model Development ===

# Decision Tree Classifier
"""
Algoritma berbasis pohon keputusan sederhana.
Membagi data berdasarkan fitur untuk mencapai klasifikasi paling bersih.
Parameter utama: criterion (gini/entropy), max_depth.
"""

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_smote, y_train_smote)

# Random Forest Classifier
"""
Ensemble learning berbasis banyak pohon keputusan.
Mengambil rata-rata prediksi untuk meningkatkan akurasi dan mencegah overfitting.
Parameter utama: n_estimators, max_depth, max_features.
"""

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_smote, y_train_smote)

# XGBoost Classifier
"""
Model boosting berbasis gradient descent.
Mengoptimalkan prediksi melalui ensemble weak learners.
Parameter utama: learning_rate, n_estimators, max_depth.
"""

xgb_model = XGBClassifier(use_label_encoder=False,
                          eval_metric='logloss', random_state=42)
xgb_model.fit(X_train_smote, y_train_smote)


# === 6. Evaluation ===
models = {
    'Decision Tree': dt_model,
    'Random Forest': rf_model,
    'XGBoost': xgb_model
}

for name, model in models.items():
    print(f"\n{name}:")
    y_pred = model.predict(X_test_scaled)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

# ROC Curve
plt.figure(figsize=(8, 6))
for name, model in models.items():
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
