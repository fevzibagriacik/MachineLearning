import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# --- 1. VERİ YÜKLEME VE HAZIRLIK ---
df = pd.read_csv("ridge_regression/house_price_regression_dataset.csv")

# X ve y ayrımı (Numpy'a çeviriyoruz)
X = df.drop(columns=["House_Price"]).to_numpy()
y = df["House_Price"].to_numpy().reshape(-1, 1)

# Normalizasyon (Ridge için ŞART, yoksa katsayılar saçmalar)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Bias (w0) için X'in başına 1'lerden oluşan sütun ekle
X = np.hstack((np.ones((X.shape[0], 1)), X))

# --- 2. TEST SETİNİ KENARA AYIRMA (BÜYÜK AYRIM) ---
# %20 Test (Kasada saklı), %80 Eğitim (K-Fold yapılacak kısım)
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. K-FOLD CROSS VALIDATION DÖNGÜSÜ ---
lambda_list = [1000, 100, 10, 1, 0.1, 0.01]
learning_rate = 0.001
epochs = 1000
kf = KFold(n_splits=5, shuffle=True, random_state=42)

lambda_scores = {} # Sonuçları buraya kaydedeceğiz

print(f"Eğitim başlıyor... (Toplam Veri: {len(X_train_full)})")

# DÖNGÜ 1: Her bir Lambda değeri için
for l in lambda_list:
    fold_errors = [] # 5 katın hatalarını tutacak
    
    # DÖNGÜ 2: K-Fold (Veriyi 5'e bölüp gezme)
    for train_index, val_index in kf.split(X_train_full):
        # İndexleri kullanarak veriyi çek
        X_train_fold, X_val_fold = X_train_full[train_index], X_train_full[val_index]
        y_train_fold, y_val_fold = y_train_full[train_index], y_train_full[val_index]
        
        # Ağırlıkları sıfırla (Her fold için sıfırdan eğitim başlar)
        m_fold, n_features = X_train_fold.shape
        w = np.zeros((n_features, 1))
        
        # DÖNGÜ 3: Gradient Descent (Eğitim)
        for epoch in range(epochs):
            # Tahmin
            y_pred = np.dot(X_train_fold, w)
            
            # Hata (Gradyan yönü için)
            error = y_train_fold - y_pred
            
            # Gradyan Hesabı (Ridge Formülü)
            grad_rss = -2 * np.dot(X_train_fold.T, error)
            grad_penalty = 2 * l * w
            grad_penalty[0] = 0 # Bias (w0) cezalandırılmaz!
            
            grad_w = (grad_rss + grad_penalty) / m_fold
            
            # Güncelleme
            w = w - learning_rate * grad_w
            
        # Bu fold bitti, Validation hatasını hesapla (MSE)
        val_pred = np.dot(X_val_fold, w)
        val_mse = np.mean((y_val_fold - val_pred) ** 2)
        fold_errors.append(val_mse)
    
    # 5 katın ortalamasını al
    avg_error = np.mean(fold_errors)
    lambda_scores[l] = avg_error
    print(f"Lambda: {l:<5} -> Ortalama Hata: {avg_error:.4f}")

# --- 4. EN İYİ MODELİ SEÇME VE FİNAL TEST ---

# Hatası en düşük olan Lambda'yı bul
best_lambda = min(lambda_scores, key=lambda_scores.get)
print("-" * 30)
print(f"🏆 Seçilen En İyi Lambda: {best_lambda}")

# Şimdi en iyi lambda ile TÜM eğitim verisini (validation ayırmadan) tekrar eğit
m_full, n_full = X_train_full.shape
final_w = np.zeros((n_full, 1))

# Final Eğitim Döngüsü
for epoch in range(2000): # Final eğitimi biraz daha uzun yapabiliriz
    y_pred = np.dot(X_train_full, final_w)
    error = y_train_full - y_pred
    
    grad_rss = -2 * np.dot(X_train_full.T, error)
    grad_penalty = 2 * best_lambda * final_w
    grad_penalty[0] = 0
    
    grad_w = (grad_rss + grad_penalty) / m_full
    final_w = final_w - learning_rate * grad_w

# --- 5. SONUÇ RAPORU (TEST SETİ) ---
test_pred = np.dot(X_test, final_w)
test_mse = np.mean((y_test - test_pred) ** 2)

print(f"🚀 Final Test Hatası (MSE): {test_mse:.4f}")
print("Final Ağırlıklar (w):\n", final_w.flatten())

# --- 6. GÖRSELLEŞTİRME (Mühendislik Analizi) ---

plt.figure(figsize=(14, 6))

# GRAFİK 1: Lambda vs Hata (Cross Validation Sonuçları)
# Amacı: Hangi lambda değerinde hatanın dip yaptığını görmek.
plt.subplot(1, 2, 1)
lambdas = list(lambda_scores.keys())
errors = list(lambda_scores.values())

plt.plot(lambdas, errors, marker='o', linestyle='-', color='b', label='CV Hatası')
plt.xscale('log') # Lambda logaritmik değiştiği için (0.01, 10, 1000) log skala şart!
plt.xlabel('Lambda (Log Scale)')
plt.ylabel('Ortalama MSE Hatası')
plt.title('Hyperparameter Tuning: En İyi Lambda Nerede?')
plt.grid(True, which="both", ls="-", alpha=0.5)

# En iyi noktayı kırmızı yıldızla işaretle
plt.scatter(best_lambda, lambda_scores[best_lambda], color='red', s=150, zorder=5, label=f'Best: {best_lambda}')
plt.legend()


# GRAFİK 2: Gerçek vs Tahmin (Prediction Alignment)
# Amacı: Noktalar kırmızı çizgi üzerinde toplanmışsa model harikadır.
plt.subplot(1, 2, 2)
plt.scatter(y_test, test_pred, alpha=0.5, color='green', edgecolors='k')

# İdeal Çizgi (y = x doğrusu)
min_val = min(np.min(y_test), np.min(test_pred))
max_val = max(np.max(y_test), np.max(test_pred))
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=3, label='İdeal Tahmin (y=x)')

plt.xlabel('Gerçek Fiyatlar (y_test)')
plt.ylabel('Tahmin Edilen Fiyatlar (y_pred)')
plt.title('Test Seti Performansı: Gerçek vs Tahmin')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()