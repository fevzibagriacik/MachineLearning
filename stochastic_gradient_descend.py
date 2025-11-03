import numpy as np
import matplotlib.pyplot as plt
import math
import random as rnd

np.random.seed(30)
x = 2 * np.random.rand(1000000)
y = 1 + 2 * x + 3 * (x ** 2) + np.random.rand(1000000)

w0 = rnd.randint(0,5)
w1 = rnd.randint(0,5)
w2 = rnd.randint(0,5)

dataset_len = len(x)

learning_rate = 0.00001 #Her seferinde sadece tek bir mini adım atacağımız için kodun patlamaması için baya küçülttük.

epochs = 5

batch_size = 1 #Her mini adım atmadan önce sadece bu kadar veri noktasına bakacağımızı belirliyoruz

cost_history = np.zeros(epochs)

current_index = 0
dataset_index = 0

for current_index in range(epochs):
    # 1 Milyonluk veri setimizin indekslerini (0'dan 999,999'a kadar)
    # her epoch'un başında rastgele karıştırıyoruz.
    indices = np.random.permutation(dataset_len)

    for i in range(0,dataset_len,batch_size): #start-stop-size
        batch_indices = indices[i : i + batch_size] #i. elemandan i+batch_size. elemana kadar olan elemanları kopyalar
        current_batch_size = len(batch_indices)

        grad_w0_total = 0.0 
        grad_w1_total = 0.0
        grad_w2_total = 0.0
    
        for dataset_index in batch_indices:
            x_i = x[dataset_index] #Karıştırdığımız ve 1 tane ayırdığımız o x değerini alır ve y ona göre hesaplanır.
            y_i = y[dataset_index]

            y_predict = (w2 * (x_i ** 2))+ (w1 * x_i) + w0 #Hata payları ve toplam gradyanlar hesaplanır
            residual = y_predict - y_i

            grad_w0_total = grad_w0_total + residual
            grad_w1_total = grad_w1_total + residual * x_i
            grad_w2_total = grad_w2_total + residual * (x_i ** 2)
    
        average_grad_w0 = (2 / current_batch_size) * grad_w0_total #Average gradyanlar hesaplanır fakat bu sefer batch_size kadar eleman vardır o yüzden ona böleriz.
        average_grad_w1 = (2 / current_batch_size) * grad_w1_total
        average_grad_w2 = (2 / current_batch_size) * grad_w2_total

        w0 = w0 - (learning_rate * average_grad_w0) #Gradyan descend formülü ile w0, w1 ve w2 değerleri hesaplanır.
        w1 = w1 - (learning_rate * average_grad_w1)
        w2 = w2 - (learning_rate * average_grad_w2) #Tekrar başa döner ve bi sonraki karışık olan diğer noktayı alarak aynı işlemleri tekrarlar.

    total_square_error = 0.0

    for dataset_index in range(10000): #Hata hesaplamaları yapılır ve cost olarak diziye yazılır.
        x_i = x[dataset_index]
        y_i = y[dataset_index]

        y_predict = (w2 * (x_i ** 2))+ (w1 * x_i) + w0
        square_error = (y_predict - y_i) ** 2
        total_square_error = total_square_error + square_error
    
    current_cost = total_square_error / dataset_len
    cost_history[current_index] = current_cost

    # Her epoch bittiğinde bilgi ver (Ne kadar hızlı gittiğimizi görmek için)
    print(f"Epoch {current_index} tamamlandı. Maliyet: {current_cost:.4f}")

print("\n--- Nihai Sonuçlar ---")
print(f"w0: {w0:.4f}")
print(f"w1: {w1:.4f}")
print(f"w2: {w2:.4f}")

# --- TALİMAT 3: Çizim Kodu (Polinom - GÜNCELLENDİ) ---

# 1. Veri Noktalarını (Scatter) Çizdirme
# 1 milyon nokta için alpha=0.01 iyi bir ayar
plt.scatter(x, y, label='Gerçek Veriler', alpha=0.01) 

# 2. Model EĞRİSİNİ Hazırlama
# Pürüzsüz bir EĞRİ çizmek için x'in min/max'ı arasında 100 nokta oluştur
x_cizim = np.linspace(np.min(x), np.max(x), 100) 

# Bu 100 x noktası için modelimizin KUADRATİK y tahminlerini hesaplayalım
# (w0, w1, w2'yi kullanarak)
y_cizim = (w2 * (x_cizim ** 2)) + (w1 * x_cizim) + w0

# 3. Model Eğrisini (Plot) Çizdirme
# Etiketi de w2'yi gösterecek şekilde güncelleyelim
plt.plot(x_cizim, y_cizim, 'r-', label=f'Bulunan Model: y = {w2:.2f}x^2 + {w1:.2f}x + {w0:.2f}', linewidth=2)

# 4. Grafiği Güzelleştirme ve Gösterme
plt.title('Gradient Descent ile Bulunan Polinomsal Uyum')
plt.xlabel('X Değerleri')
plt.ylabel('Y Değerleri')
plt.legend() 
plt.grid(True) 
plt.show()

# --- TALİMAT 4: Maliyet Geçmişi Grafiği ---

plt.figure(figsize=(10, 6)) # Grafiğin boyutunu ayarla
plt.plot(cost_history, label='Maliyet (MSE) Geçmişi', color='blue') # Maliyet geçmişini çizdir

plt.title('Maliyet Fonksiyonunun (MSE) Zaman İçindeki Değişimi') # Başlık
plt.xlabel('Epoch (Adım Sayısı)') # X ekseni etiketi
plt.ylabel('Maliyet (MSE)') # Y ekseni etiketi
plt.legend() # Etiket kutusunu göster
plt.grid(True) # Izgara ekle
plt.show() # Grafiği ekranda göster

#Mini batch'ten tek farkı batch size değerinin 1 olmasıdır.