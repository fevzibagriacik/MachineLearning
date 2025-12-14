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

lr = 0.001
epochs = 1000
lambda_ = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]

#Slice k-fold
fold_number = 10
fold_size = int(sample_number / fold_number)
cv_errors_matrix = np.zeros((len(lambda_), fold_number))

for i in range(fold_number):
    start_index = i * fold_size
    end_index = (i+1) * fold_size

    X_val = X_shuffled[start_index : end_index]
    y_val = y_shuffled[start_index : end_index]

    X_train_part1 = X_shuffled[:start_index]
    X_train_part2 = X_shuffled[end_index:]
    X_train = np.concatenate((X_train_part1, X_train_part2), axis=0)

    y_train_part1 = y_shuffled[:start_index]
    y_train_part2 = y_shuffled[end_index:]
    y_train = np.concatenate((y_train_part1, y_train_part2), axis=0)

    #Gradient Descent
    for current_lambda in range(len(lambda_)):
        w = np.zeros((feature_number, 1))

        for epoch in range(epochs):
            y_pred = np.dot(X_train, w)

            error = y_pred - y_train

            train_sample_number = X_train.shape[0]
            grad_w = (2/train_sample_number) * np.dot(X_train.T, error)

            w_penalty = w.copy()
            w_penalty[0] = 0
            grad_penalty = lambda_[current_lambda] * np.sign(w_penalty)

            w = w - lr * (grad_w + grad_penalty)
        
        #Validation set
        y_val_pred = np.dot(X_val, w)
        val_error = y_val_pred - y_val
        mse_val = np.mean(val_error ** 2)

        cv_errors_matrix[current_lambda, i] = mse_val
    
average_cv_errors = np.mean(cv_errors_matrix, axis=1)

# --- 3. EN İYİ LAMBDA SEÇİMİ ---
best_idx = np.argmin(average_cv_errors)
best_lambda = lambda_[best_idx]
min_error = average_cv_errors[best_idx]
best_lambda_fold_errors = cv_errors_matrix[best_idx, :]

print(f"✅ En İyi Lambda Bulundu: {best_lambda}")

# --- 4. FİNAL MODEL EĞİTİMİ (TÜM VERİ İLE) ---
print("2. Aşama: Final Model Eğitiliyor...")

w_final = np.zeros((feature_number, 1))
final_lr = 0.001 
final_epochs = 2000 

for epoch in range(final_epochs):
    y_pred = np.dot(X_shuffled, w_final)
    error = y_pred - y_shuffled
    grad_w = (2/sample_number) * np.dot(X_shuffled.T, error)
    
    w_penalty = w_final.copy(); w_penalty[0] = 0
    grad_penalty = lambda_[current_lambda] * np.sign(w_penalty)
    
    w_final = w_final - final_lr * (grad_w + grad_penalty)

y_final_pred = np.dot(X_shuffled, w_final)

# --- 5. GRAFİK ÇİZİMİ (3 GRAFİK BİR ARADA) ---
plt.figure(figsize=(14, 10)) # Büyük bir pencere

# GRAFİK 1 (SOL ÜST): Lambda Seçimi
plt.subplot(2, 2, 1)
plt.semilogx(lambda_, average_cv_errors, marker='o', linestyle='-', color='blue', linewidth=2)
plt.scatter(best_lambda, min_error, color='red', s=150, marker='*', zorder=5, label=f'En İyi: {best_lambda}')
plt.title('1. Model Seçimi (Lambda vs Hata)')
plt.xlabel('Lambda (Log Scale)')
plt.ylabel('Ortalama CV Hatası')
plt.grid(True, which="both", ls="-", alpha=0.4)
plt.legend()

# GRAFİK 2 (SAĞ ÜST): Fold Kararlılığı
plt.subplot(2, 2, 2)
fold_indices = np.arange(1, fold_number + 1)
bars = plt.bar(fold_indices, best_lambda_fold_errors, color='mediumseagreen', edgecolor='black', alpha=0.7)
plt.axhline(y=min_error, color='red', linestyle='--', label=f'Ort: {min_error:.3f}')
plt.title(f'2. Kararlılık Analizi (Lambda={best_lambda})')
plt.xlabel('Fold Numarası')
plt.ylabel('Hata')
plt.xticks(fold_indices)
plt.legend()
plt.grid(True, axis='y', alpha=0.4)

# GRAFİK 3 (ALT TARAF - GENİŞ): Gerçek vs Tahmin
plt.subplot(2, 1, 2) # 2 satır, 1 sütunluk yerin 2.sini (altını) kapla
plt.scatter(y_shuffled, y_final_pred, color='royalblue', alpha=0.6, label='Veri Noktaları')

# İdeal Çizgi
min_v = min(np.min(y_shuffled), np.min(y_final_pred)) - 0.5
max_v = max(np.max(y_shuffled), np.max(y_final_pred)) + 0.5
plt.plot([min_v, max_v], [min_v, max_v], color='red', linestyle='--', linewidth=3, label='Mükemmel Tahmin Çizgisi')

plt.title(f'3. Nihai Model Performansı (Tüm Veri Seti)\nLambda={best_lambda}')
plt.xlabel('Gerçek Fiyatlar (Normalize)')
plt.ylabel('Tahmin Edilen Fiyatlar (Normalize)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("Tüm analiz grafikleri çizildi.")