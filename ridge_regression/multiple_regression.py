import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

df = pd.read_csv('house_price_regression_dataset.csv')

X = df[['Square_Footage','Num_Bedrooms','Num_Bathrooms','Year_Built','Lot_Size','Garage_Size','Neighborhood_Quality']].values
y = df[['House_Price']].values

# --- DÜZELTME BAŞLANGICI: NORMALİZASYON ---
# Gradyan inişinin düzgün çalışması için verileri ölçekliyoruz (Z-score normalization)
# (Değer - Ortalama) / Standart Sapma
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean) / X_std

y_mean = np.mean(y, axis=0)
y_std = np.std(y, axis=0)
y = (y - y_mean) / y_std
# --- DÜZELTME BİTİŞİ ---

sample_number = X.shape[0] #sample number

ones_column = np.ones((sample_number, 1)) #for w0, we added a column includes 1
X_bias = np.hstack((ones_column, X)) #ones_column is moved to left of X (20x8)

feature_number = X_bias.shape[1] #column number with added ones_column

w = np.zeros((feature_number, 1)) #(8x1)

np.random.seed(42)
rnd_index = np.random.permutation(sample_number) #Shuffle index
X_shuffled = X_bias[rnd_index] #randomize X matrix
y_shuffled = y[rnd_index] #randomize y matrix

#Slice dataset
train_ratio = 0.7
test_ratio = 0.15
train_limit = int(sample_number * train_ratio) 
test_limit = train_limit + int(sample_number * test_ratio)
X_train = X_shuffled[:train_limit]
y_train = y_shuffled[:train_limit]
X_validate = X_shuffled[train_limit : test_limit]
y_validate = y_shuffled[train_limit : test_limit]
X_test = X_shuffled[test_limit:]
y_test = y_shuffled[test_limit:]

lr = 0.001
epochs = 1000

train_cost_history = []
val_cost_history = []
test_cost_history = []

for epoch in range(epochs):
    #Gradient Descent
    y_pred = np.dot(X_train, w)

    error = y_pred - y_train
    train_mse = np.mean(error ** 2)

    train_cost_history.append(train_mse)

    train_sample_number = X_train.shape[0]
    grad_w = (2/train_sample_number) * np.dot(X_train.T, error)

    w = w - lr * grad_w

#Validation set
y_val_pred = np.dot(X_validate, w)
val_error = y_val_pred - y_validate
val_mse = np.mean(val_error ** 2)
val_cost_history.append(val_mse)

#Test set
y_test_pred = np.dot(X_test, w)
test_error = y_test_pred - y_test
test_mse = np.mean(test_error ** 2)
test_cost_history.append(test_mse)






# --- 4'LÜ GRAFİK ÇİZİMİ ---
plt.figure(figsize=(14, 10)) # Geniş ve Yüksek bir pencere

# 1. SOL ÜST: Loss History (Train vs Val)
plt.subplot(2, 2, 1)
plt.plot(train_cost_history, color='blue', label='Training Loss', linewidth=2)
plt.plot(val_cost_history, color='orange', label='Validation Loss', linewidth=2, linestyle='--')
plt.title('1. Öğrenme Eğrisi (Overfitting Kontrolü)')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)

# 2. SAĞ ÜST: Feature Importance (Ağırlıklar)
plt.subplot(2, 2, 2)
feature_names = ['Bias', 'Sq_Ft', 'Bed', 'Bath', 'Year', 'Lot', 'Garage', 'Qual']
weights = w.flatten()
colors = ['red' if x < 0 else 'green' for x in weights]
plt.barh(feature_names, weights, color=colors)
plt.axvline(0, color='black', linewidth=1)
plt.title('2. Hangi Özellik Fiyatı Etkiliyor?')
plt.grid(True, axis='x')

# 3. SOL ALT: Validation Seti Başarısı
y_val_final = np.dot(X_validate, w) # Son ağırlıklarla tahmin
plt.subplot(2, 2, 3)
plt.scatter(y_validate, y_val_final, color='orange', alpha=0.7, label='Val Verisi')
# İdeal çizgi
min_v = min(np.min(y_validate), np.min(y_val_final)) - 0.5
max_v = max(np.max(y_validate), np.max(y_val_final)) + 0.5
plt.plot([min_v, max_v], [min_v, max_v], color='red', linestyle='--', linewidth=3)
plt.title(f'3. Validation Başarısı (Ara Sınav)\nMSE: {val_cost_history[-1]:.4f}')
plt.xlabel('Gerçek')
plt.ylabel('Tahmin')
plt.grid(True)

# 4. SAĞ ALT: Test Seti Başarısı
plt.subplot(2, 2, 4)
plt.scatter(y_test, y_test_pred, color='purple', alpha=0.7, label='Test Verisi')
# İdeal çizgi
min_t = min(np.min(y_test), np.min(y_test_pred)) - 0.5
max_t = max(np.max(y_test), np.max(y_test_pred)) + 0.5
plt.plot([min_t, max_t], [min_t, max_t], color='red', linestyle='--', linewidth=3)
plt.title(f'4. Test Başarısı (Final Sınavı)\nMSE: {test_mse:.4f}')
plt.xlabel('Gerçek')
plt.ylabel('Tahmin')
plt.grid(True)

plt.tight_layout()
plt.show()

# SONUÇLARI YAZDIRMA
print("-" * 30)
print(f"Training MSE   : {train_cost_history[-1]:.5f}")
print(f"Validation MSE : {val_cost_history[-1]:.5f}")
print(f"TEST MSE       : {test_mse:.5f}")
print("-" * 30)