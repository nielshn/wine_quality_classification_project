# Import Libraries
from sklearn.metrics import ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV


# Load Datasets
red = pd.read_csv("winequality-red.csv", sep=';')
white = pd.read_csv("winequality-white.csv", sep=';')
red["type"], white["type"] = "red", "white"
df = pd.concat([red, white], axis=0)

# Binary target: 0 (low: 3–6), 1 (high: 7–9)
df['quality_label'] = df['quality'].apply(lambda q: 1 if q >= 7 else 0)

# ==== Exploratory Data Analysis (EDA) ====
# Distribusi label
plt.figure(figsize=(6, 4))
sns.countplot(x='quality_label', hue='type', data=df)
plt.title("Distribusi Kualitas Wine")
plt.xlabel("Label Kualitas (0 = Rendah, 1 = Tinggi)")
plt.tight_layout()
plt.show()

# Korelasi fitur
plt.figure(figsize=(12, 10))
sns.heatmap(df.drop(['quality', 'type'], axis=1).corr(),
            annot=True, cmap='coolwarm')
plt.title("Korelasi antar Fitur")
plt.tight_layout()
plt.show()

# Boxplot sebelum outlier removal
features = df.columns[:-3]
plt.figure(figsize=(20, 15))
for i, feature in enumerate(features):
    plt.subplot(4, 3, i+1)
    sns.boxplot(x='quality_label', y=feature, data=df)
    plt.title(f"{feature} vs Kualitas")
plt.tight_layout()
plt.show()

# ==== Outlier Removal ====


def remove_outliers(df, features):
    df_clean = df.copy()
    for feature in features:
        Q1 = df_clean[feature].quantile(0.25)
        Q3 = df_clean[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[feature] >= lower_bound)
                            & (df_clean[feature] <= upper_bound)]
    return df_clean


df_clean = remove_outliers(df, features)

# ==== Preprocessing ====
X = df_clean.drop(['quality', 'quality_label', 'type'], axis=1)
y = df_clean['quality_label']

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# ==== SMOTE ====
sm = SMOTE(random_state=42)
X_train_smote, y_train_smote = sm.fit_resample(X_train_scaled, y_train)

# ==== Modeling ====

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_smote, y_train_smote)
print("DT Accuracy:", accuracy_score(y_test, dt.predict(X_test_scaled)))

# Random Forest (Randomized Search)
rf = RandomForestClassifier(random_state=42)
param_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt']
}
rf_search = RandomizedSearchCV(
    rf, param_rf, n_iter=10, cv=5,
    scoring='accuracy', n_jobs=-1, random_state=42)
rf_search.fit(X_train_smote, y_train_smote)
rf_best = rf_search.best_estimator_
print("Random Forest (Tuned) Accuracy:", accuracy_score(
    y_test, rf_best.predict(X_test_scaled)))

# SVM (Grid Search)
svm = SVC(probability=True)
svm_param = {'C': [0.1, 1, 10], 'gamma': ['scale', 0.01]}
svm_search = GridSearchCV(svm, svm_param, cv=5, scoring='accuracy', n_jobs=-1)
svm_search.fit(X_train_smote, y_train_smote)
svm_best = svm_search.best_estimator_
print("SVM Accuracy:", accuracy_score(y_test, svm_best.predict(X_test_scaled)))

# XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train_smote, y_train_smote)
print("XGBoost Accuracy:", accuracy_score(y_test, xgb.predict(X_test_scaled)))


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"\n===== {name} =====")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=[
                           "Low Quality", "High Quality"]).plot(cmap='Blues')
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.show()


evaluate_model("Decision Tree", dt, X_test_scaled, y_test)
evaluate_model("Random Forest", rf_best, X_test_scaled, y_test)
evaluate_model("SVM", svm_best, X_test_scaled, y_test)
evaluate_model("XGBoost", xgb, X_test_scaled, y_test)


# ==== ROC Curve (Random Forest) ====
y_proba_rf = rf_best.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba_rf)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(
    fpr, tpr, label=f"Random Forest AUC = {roc_auc:.2f}", color='darkorange')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
