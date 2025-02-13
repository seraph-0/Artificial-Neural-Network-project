import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Örnek veri kümesini yükleme (Iris Dataset)
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

# Veri kümesini eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# SVM (Support Vector Machine) ile sınıflandırma
svm_model = SVC(kernel='linear')  # Linear kernel kullanıyoruz
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

# KNN (K-Nearest Neighbors) ile sınıflandırma
knn_model = KNeighborsClassifier(n_neighbors=3)  # K=3
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

# Karışıklık Matrisini Görselleştiren Fonksiyon
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=data.target_names, yticklabels=data.target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.show()

# SVM Modelinin Performansını Yazdırma
print("SVM Classification Report:")
print(classification_report(y_test, y_pred_svm))

# KNN Modelinin Performansını Yazdırma
print("KNN Classification Report:")
print(classification_report(y_test, y_pred_knn))

# Karışıklık Matrisi Çizimi
plot_confusion_matrix(y_test, y_pred_svm, "Confusion Matrix for SVM")
plot_confusion_matrix(y_test, y_pred_knn, "Confusion Matrix for KNN")

# Veri Kümesinde Bazı Değerleri NaN ile Değiştirme (örneğin, %10 oranında)
np.random.seed(42)  # Sabit rastgelelikin sağlanması için
nan_indices = np.random.rand(*X.shape) < 0.1  # %10 değerleri NaN yapmak için rastgele indeksler
X_nan = X.mask(nan_indices)  # İlgili değerleri NaN ile değiştir

print(X_nan.head())

# NaN Değerlerini, Her Sütunun Ortalamasıyla Doldurma
X_filled = X_nan.fillna(X_nan.mean())

# NaN'lar ile doldurulmuş veri kümesinin eğitim ve test setlerine ayırma
X_train_filled, X_test_filled, y_train_filled, y_test_filled = train_test_split(X_filled, y, test_size=0.3, random_state=42)

# NaN ile doldurulmuş veri kümesiyle SVM ve KNN Sınıflandırma
svm_model_filled = SVC(kernel='linear')  # Linear kernel kullanıyoruz
svm_model_filled.fit(X_train_filled, y_train_filled)
y_pred_svm_filled = svm_model_filled.predict(X_test_filled)

knn_model_filled = KNeighborsClassifier(n_neighbors=3)  # K=3
knn_model_filled.fit(X_train_filled, y_train_filled)
y_pred_knn_filled = knn_model_filled.predict(X_test_filled)

# NaN ile doldurulmuş veri kümesinin Karışıklık Matrislerini Çizme
print("\nSVM Classification Report after NaN Replacement:")
print(classification_report(y_test_filled, y_pred_svm_filled))

print("\nKNN Classification Report after NaN Replacement:")
print(classification_report(y_test_filled, y_pred_knn_filled))

# Karışıklık Matrisi Çizimi
plot_confusion_matrix(y_test_filled, y_pred_svm_filled, "Confusion Matrix for SVM after NaN Replacement")
plot_confusion_matrix(y_test_filled, y_pred_knn_filled, "Confusion Matrix for KNN after NaN Replacement")
