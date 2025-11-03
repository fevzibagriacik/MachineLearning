import numpy as np
import matplotlib.pyplot as plt
import math

#Sahte data set ve formülü

np.random.seed(42)
x = 2 * np.random.rand(20)
y = 1 + 2 * x + np.random.randn(20)

dataset_len = len(x)

plt.scatter(x,y)
plt.title("Sample dataset")
plt.xlabel("x value")
plt.ylabel("y value")
plt.grid(True)
plt.show()

#Formülü bilmiyormuş gibi brute force ile bulmaya çalışmak

w0_range = np.linspace(-5,5,100)
w1_range = np.linspace(-5,5,100)

cost_min = math.inf
best_w0 = None
best_w1 = None

cost_grid = np.zeros((len(w0_range), len(w1_range)))

for i, w0 in enumerate(w0_range):
    for j, w1 in enumerate(w1_range):
        total_error = 0
        for dataset_index in range(dataset_len):
            x_i = x[dataset_index]
            y_i = y[dataset_index]

            y_predict = w1 * x_i + w0
            
            error_square = (y_i - y_predict) ** 2

            total_error = total_error + error_square
        
        #Ortalama costu(MSE) hesapla
        cost = total_error / dataset_len
        
        #Bu costu çizim için gride kaydet
        cost_grid[i, j] = cost

        #Minimum costu güncelle
        if cost < cost_min:
            cost_min = cost
            best_w0 = w0
            best_w1 = w1

print(f"Bulunan en düşük cost(MSE): {cost_min:.4f}")
print(f"En iyi w0: {best_w0:.4f}")
print(f"En iyi w1: {best_w1:.4f}")

#Sonucu çizme

plt.figure(figsize=(10,7))
plt.contourf(w1_range, w0_range, cost_grid)
plt.colorbar(label="Cost(MSE)")
plt.plot(best_w1, best_w0, 'rx', markersize=15, label=f'En İyi Nokta\nw0={best_w0:.2f}, w1={best_w1:.2f}')
plt.xlabel('w1 (Eğim / Weight)')
plt.ylabel('w0 (Bias / Intercept)')
plt.title('Brute Force ile Taranan Maliyet Yüzeyi')
plt.legend()
plt.grid(True)
plt.show()

# İlk grafik kapatıldıktan sonra, program devam eder ve 
# bu komutla İKİNCİ bir boş tuval açar.
plt.figure(figsize=(10, 7)) 

# En başta oluşturduğumuz o 20 adet (x, y) veri noktamızı 
# mavi noktalar (scatter) halinde çizer.
plt.scatter(x, y, label='Gerçek Veriler') 

# Düz bir çizgi çizmek için sadece iki uç noktaya ihtiyacımız var.
# Bu komut, 'x' dizimizdeki en küçük değeri (örn: 0.1) ve en büyük değeri
# (örn: 1.9) bulup [0.1, 1.9] şeklinde yeni bir mini dizi oluşturur.
x_cizim = np.array([np.min(x), np.max(x)]) 

# Burada, brute force ile bulduğumuz best_w1 ve best_w0 değerlerini kullanarak
# y = (best_w1 * x) + best_w0 formülünü uygularız.
# Numpy sayesinde bu, x_cizim içindeki *iki nokta için de* (en küçük x 
# ve en büyük x) aynı anda yapılır. Sonuç olarak y_cizim, bu iki x 
# noktasına karşılık gelen tahmin edilen y değerlerini tutar.
y_cizim = best_w1 * x_cizim + best_w0 

# Bu komut, noktaları değil, bir çizgi çizer.
# 1. (x_cizim, y_cizim): Çizginin uç noktalarını alır.
# 2. 'r-': Rengin 'r' (Red/Kırmızı) ve çizgi stilinin '-' (düz çizgi) 
#    olmasını söyler.
# 3. label=...: Açıklama kutusu için bir etiket ekler. f-string kullanarak
#    formülü bulduğumuz değerlerle yazarız.
# 4. linewidth=3: Çizgiyi biraz kalın (3 piksel) yaparak daha görünür hale getirir.
plt.plot(x_cizim, y_cizim, 'r-', label=f'Bulunan Model\ny = {best_w1:.2f}x + {best_w0:.2f}', linewidth=3)

# Tıpkı ilk grafikteki gibi, bu grafiğe de başlıklar, etiketler ekler.
plt.xlabel('X Değeri')
plt.ylabel('y Değeri')
plt.title('Brute Force ile Bulunan En İyi Uyum Çizgisi')
plt.legend() # Açıklama kutusunu (Gerçek Veriler, Bulunan Model) gösterir.
plt.grid(True)

# Ve son olarak, hazırlanan bu ikinci grafiği de ekranda gösterir.
plt.show()

#SUMMARY
#1-Veri seti oluşturuldu, ve gizli bir formül verdik; bu formülü bilmiyormuş gibi davrandık.
#Verdiğimiz değerler w0 = 1 ve w1 = 2 idi.

#2-x değerlerini np.random.rand(20) ile 0 ile 20 arasında 20 adet rastgele x değeri oluşturduk
#ve bunlar numpy dizisi içerisinde saklandı.

#3-y değerlerini bu dizi içerisinden yerine koyarak değerlerini çektik ve üzerine küçük bir sapma
#ekledik gerçekçi görünebilmesi için.

#4-Verileri ekranda gösterdik.

#5-np.linspace(-5,5,100) komutuyla hem w0 hem de w1 için -5 ile 5 arasında eşit aralıklı 100'er
#adet aday değer oluşturduk.

#6-100x100lük boş bir cost matrixi oluşturduk.

#7-Döngülerle tek tek değerleri gezdik ve w0 ile w1 değerlerini tek tek aldık. Bunları tahmin
#denklemine yerleştirdik ve her bir değer için bulduğumuz hata karelerini toplam hataya ekleyerek
#toplam hatayı bir denklem için bulmuş olduk.

#8-Toplam hatayı 20'ye bölerek ortalama costu bulduk ve bunu boş matrixe kaydettik.

#9-Minimum cost varsa bunu güncelledik ve w0 ile w1 değerlerini tuttuk.

#10-Döngü şeklinde değerleri bulmaya devam ettik ve matrixi tamamen doldurduk.

#11-Renklendirme ile maliyetlerin büyük ve küçüklüğünü gösterdik. En koyu yer ise bizim minimum
#costumuz oldu.

#12-En iyi w0 ve w1 değerleri ile çizgimizi çizdirdik ve değerler tam tutmasa da başlangıçtaki
#değerleri yakın çıktı.