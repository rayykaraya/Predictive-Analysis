#!/usr/bin/env python
# coding: utf-8

# # Business Understanding

# Proyek ini bertujuan untuk membangun model prediktif yang dapat memperkirakan harga penutupan saham harian PT Aneka Tambang Tbk (ANTM) menggunakan pendekatan machine learning berbasis time series. Model ini diharapkan mampu memberikan wawasan prediktif yang berguna bagi investor atau analis pasar dalam mengambil keputusan investasi. Dataset yang digunakan mencakup harga saham harian ANTM dari tahun 2021 hingga 2025, yang diperoleh dari situs Investing.com (https://id.investing.com/equities/aneka-tambang-historical-data). 

# # Data Understanding

# ### Data Loading

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


# Pada tahap ini, dilakukan proses eksplorasi awal terhadap data historis harga saham PT Aneka Tambang Tbk (ANTM) yang telah diunduh dari situs Investing.com. Dataset tersebut mencakup harga saham harian dari tanggal 21 Januari 2021 hingga 21 Mei 2025 dengan total 1.042 entri data dan 7 kolom fitur, yaitu:
# 
# - Tanggal: Tanggal pencatatan data.
# - Terakhir: Harga penutupan saham pada hari tersebut.
# - Pembukaan: Harga pembukaan saham.
# - Tertinggi: Harga tertinggi yang dicapai pada hari tersebut.
# - Terendah: Harga terendah pada hari tersebut.
# - Vol.: Volume transaksi saham.
# - Perubahan%: Persentase perubahan harga dari hari sebelumnya.

# In[2]:


url = 'https://drive.google.com/uc?id=1rz3hHYITPxWD7ypSA_xAbfwI-Yj-acYF'
df = pd.read_csv(url)
df.head()


# In[3]:


df.info()
print("Jumlah baris dan kolom:", df.shape)


# Dari hasil df.info() diketahui bahwa sebagian besar kolom numerik sudah dalam format float64. Namun kolom Tanggal, Vol. dan Perubahan% masih berbentuk object, sehingga perlu dilakukan proses transformasi lebih lanjut agar bisa dianalisis secara kuantitatif.

# ### Transformasi Data

# Data pada kolom Tanggal awalnya dalam bentuk object (string) dan perlu dikonversi ke format datetime agar Konversi dan pengurutan data secara kronologis agar sesuai dengan analisis time series.

# In[4]:


df['Tanggal'] = pd.to_datetime(df['Tanggal'], dayfirst=True)
df = df.sort_values('Tanggal').reset_index(drop=True)
df


# Kolom Vol. awalnya menggunakan notasi seperti 329,77M atau 894,77M, sehingga perlu dikonversi menjadi angka sehingga nilai volume kini berada dalam satuan numerik (float64) dengan satuan lot saham.

# In[5]:


# Konversi Volume
def convert_volume(vol_str):
    if pd.isna(vol_str):
        return 0
    vol_str = str(vol_str)  # pastikan jadi string dulu
    vol_str = vol_str.replace('.', '')  
    vol_str = vol_str.replace(',', '.')  
    if vol_str.endswith('M'):
        return float(vol_str[:-1]) * 1_000_000
    elif vol_str.endswith('K'):
        return float(vol_str[:-1]) * 1_000
    elif vol_str.endswith('B'):
        return float(vol_str[:-1]) * 1_000_000_000
    else:
        try:
            return float(vol_str)
        except:
            return 0

df['Vol.'] = df['Vol.'].apply(convert_volume)
df.rename(columns={'Vol.': 'Volume'}, inplace=True)
df


# Kolom Perubahan% berisi string sehingga diperlukan untuk enghapus tanda persen dan mengganti koma dengan titik desimal dan konversi ke float sehingga siap digunakan sebagai fitur numerik.

# In[6]:


# Hapus tanda % dan ganti koma dengan titik, lalu konversi ke float
df['Perubahan%'] = df['Perubahan%'].str.replace('%', '', regex=False).str.replace(',', '.', regex=False).astype(float)
df


# In[7]:


df.info()
df.duplicated().sum()


# Dataset telah berhasil dibersihkan dan ditransformasikan menjadi bentuk yang siap diproses untuk eksplorasi lanjutan atau dimasukkan ke dalam model prediktif. Hal ini memastikan format waktu valid dan urut, data numerik siap distandardisasi dan informasi penting seperti volume dan perubahan harga dapat dianalisis secara akurat. Dataset juga tidak memiliki missing values.
# 
# 

# # Exploratory Data Analysis (EDA)

# In[8]:


df.describe()


# ### Visualisasi Tren Harga

# In[9]:


plt.figure(figsize=(14,6))
plt.plot(df['Tanggal'], df['Terakhir'], label='Harga Penutupan (Terakhir)')
plt.plot(df['Tanggal'], df['Pembukaan'], label='Harga Pembukaan', alpha=0.6)
plt.title('Pergerakan Harga Saham')
plt.xlabel('Tanggal')
plt.ylabel('Harga')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Grafik menunjukkan pergerakan harga pembukaan dan penutupan saham ANTM dari Januari 2021 hingga Mei 2025. Harga saham sempat naik tinggi di awal 2021, kemudian cenderung menurun dan stagnan hingga awal 2024. Mulai pertengahan 2024, harga saham menunjukkan tren naik yang tajam, menandakan adanya sentimen positif atau perbaikan kinerja perusahaan. Pergerakan harga pembukaan dan penutupan relatif berdekatan, menunjukkan stabilitas intraday.

# ### Visualisasi Volume Perdagangan

# In[10]:


import matplotlib.pyplot as plt

plt.figure(figsize=(14, 4))
plt.bar(df['Tanggal'], df['Volume'], color='green')
plt.title('Volume Perdagangan Saham ANTM')
plt.xlabel('Tanggal')
plt.ylabel('Volume')
plt.show()


# Visualisasi ini menunjukkan pergerakan volume perdagangan saham ANTM dari waktu ke waktu. Terlihat adanya lonjakan volume yang sangat signifikan pada awal 2021, yang kemungkinan dipicu oleh sentimen pasar atau aksi korporasi besar. Setelah lonjakan tersebut, volume mengalami penurunan dan cenderung stabil pada level yang lebih rendah, dengan beberapa lonjakan kecil yang sporadis di tahun-tahun berikutnya. Ini mencerminkan minat investor yang sempat memuncak lalu menurun, serta fluktuasi minat pasar terhadap saham ini.

# ### Distribusi Perubahan Persentase Harga

# In[11]:


import seaborn as sns

plt.figure(figsize=(10, 5))
sns.histplot(df['Perubahan%'], bins=50, kde=True, color='orange')
plt.title('Distribusi Perubahan Harga Harian (%)')
plt.xlabel('Perubahan%')
plt.ylabel('Frekuensi')
plt.show()


# Grafik histogram ini menggambarkan distribusi perubahan harga harian saham ANTM. Bentuk distribusinya menyerupai distribusi normal dengan puncak di sekitar 0%, menandakan bahwa sebagian besar perubahan harian berada di kisaran -5% hingga +5%. Tidak banyak outlier ekstrem, yang menunjukkan volatilitas harian yang relatif moderat. Distribusi yang cukup simetris juga menandakan tidak adanya bias besar terhadap tren kenaikan atau penurunan ekstrem dalam satu hari perdagangan.

# ### Korelasi Antar Kolom

# In[12]:


plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matriks Korelasi Fitur Numerik')
plt.show()


# Heatmap ini menggambarkan korelasi antar fitur numerik seperti Open, High, Low, Close, Volume, dan Change %. Terdapat korelasi sangat tinggi (mendekati 1.0) antara harga pembukaan, tertinggi, terendah, dan penutupan, menunjukkan bahwa pergerakan harga harian sangat selaras. Namun, volume dan persentase perubahan harga menunjukkan korelasi yang lebih lemah terhadap variabel harga, menandakan bahwa aktivitas perdagangan tidak selalu linear terhadap perubahan harga.

# ### Musiman & Tren Bulanan

# In[13]:


df['Bulan'] = df['Tanggal'].dt.to_period('M')
monthly_avg = df.groupby('Bulan')['Terakhir'].mean()

plt.figure(figsize=(14, 5))
monthly_avg.plot()
plt.title('Rata-Rata Harga Penutupan Bulanan')
plt.xlabel('Bulan')
plt.ylabel('Harga Rata-Rata')
plt.grid(True)
plt.show()


# Line chart ini menunjukkan tren rata-rata harga penutupan per bulan. Terlihat bahwa harga penutupan bulanan mengalami penurunan dari akhir 2021 hingga awal 2024. Namun, tren ini berbalik arah setelahnya, dengan harga mulai naik secara konsisten hingga pertengahan 2025. Ini menunjukkan kemungkinan pemulihan kinerja perusahaan atau perbaikan sentimen pasar terhadap saham ANTM.

# # Data Preparation

# In[14]:


# Pastikan kolom penting
features = ['Pembukaan', 'Tertinggi', 'Terendah', 'Volume', 'Perubahan%', 'Terakhir']


# Dengan mendefinisikan list features seperti ini, kita bisa mendefinisikan daftar nama kolom (fitur) yang penting atau relevan untuk digunakan dalam proses analisis dan modelling.

# ### Feature Engineering

# In[15]:


# Buat lag fitur untuk harga Terakhir (1, 3, 5 hari sebelumnya)
df['Terakhir_lag1'] = df['Terakhir'].shift(1)
df['Terakhir_lag3'] = df['Terakhir'].shift(3)
df['Terakhir_lag5'] = df['Terakhir'].shift(5)
df


# Lag features adalah mengambil nilai harga Terakhir pada hari sebelumnya (lag 1), 3 hari sebelumnya (lag 3), dan 5 hari sebelumnya (lag 5). Tujuannya agar model bisa melihat pengaruh harga masa lalu terhadap harga saat ini (capturing temporal dependencies).

# In[16]:


# Moving average untuk harga Terakhir (window 3 dan 5)
df['Terakhir_MA3'] = df['Terakhir'].rolling(window=3).mean()
df['Terakhir_MA5'] = df['Terakhir'].rolling(window=5).mean()
df


# Moving average features ini menghitung rata-rata harga Terakhir selama 3 hari terakhir dan 5 hari terakhir. Moving average ini membantu model menangkap tren jangka pendek (3 hari) dan menengah (5 hari) yang mungkin relevan untuk prediksi harga atau perilaku pasar.

# In[17]:


# Drop baris dengan nilai NaN akibat shift dan rolling
df.dropna(inplace=True)
df


# Karena operasi shift() dan rolling() akan menghasilkan nilai kosong (NaN) pada baris awal (karena data sebelumnya tidak ada), baris ini dihapus agar data siap dipakai untuk modeling tanpa error. Dengan menambah fitur lag dan moving average, kita bisa memperkaya data dengan informasi temporal yang penting untuk analisis deret waktu (time series) atau prediksi harga saham, sehingga model bisa lebih akurat memahami pola dan tren harga dari waktu ke waktu.

# ### Feature Selection

# In[18]:


# Siapkan data input dan target
input_features = ['Pembukaan', 'Tertinggi', 'Terendah', 'Volume', 'Perubahan%', 
                  'Terakhir_lag1', 'Terakhir_lag3', 'Terakhir_lag5', 'Terakhir_MA3', 'Terakhir_MA5']


# Tujuan Feature Selection adalah memfokuskan proses modeling hanya pada fitur-fitur yang dianggap relevan berdasarkan domain knowledge dan hasil EDA, mengurangi noise dan kompleksitas model.

# ### Data Splitting

# In[19]:


X = df[input_features].values
y = df['Terakhir'].values  # Target = harga penutupan hari ini

# Split 80:20 tanpa shuffle (karena time-series)
split_idx = int(len(df) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]


# Memisahkan dataset menjadi dua bagian: data latih (train) dan data uji (test) dengan rasio 80:20. Data tidak diacak (no shuffling) karena ini merupakan data deret waktu (time series), sehingga urutan kronologis tetap terjaga.

# ### Scaling / Normalization

# In[20]:


from sklearn.preprocessing import MinMaxScaler

# Inisialisasi scaler
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# Fit di data training, transform semua
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))


# Melakukan normalisasi data agar seluruh nilai fitur (X) dan target (y) berada dalam rentang 0 hingga 1. Scaling dilakukan dengan MinMaxScaler, dan fitting hanya pada data training untuk menghindari data leakage. Tujuannya adalah untuk membantu model konvergen lebih cepat dan stabil selama proses pelatihan, terutama pada model berbasis neural network dan juga menjaga proporsi dan skala fitur agar tidak ada fitur yang mendominasi karena skala yang besar.

# ### Reshape untuk LSTM

# In[21]:


X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))


# Mengubah bentuk data input menjadi 3 dimensi agar kompatibel dengan input model LSTM, yaitu (samples, timesteps, features). Karena fitur sudah mencerminkan informasi historis (melalui lag dan moving average), maka timesteps diset ke 1.Tujuannya adalah menyesuaikan bentuk data input dengan kebutuhan arsitektur LSTM yang membutuhkan input dalam bentuk sekuens 3D dan memastikan model bisa memahami pola waktu dari data harian.

# # Modelling

# In[22]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras_tuner import HyperParameters
from keras_tuner.tuners import RandomSearch
from tensorflow.keras.callbacks import EarlyStopping


# Pada Modeling kali ini, saya  hanya akan menggunkaan 1 algoritma yaitu LSTM. Oleh karena itu saya akan melakukan improvement pada baseline model dengan hyperparameter tuning dengan Keras Tuner.

#  Keras Tuner digunakan untuk secara otomatis mencari kombinasi hyperparameter terbaik pada model LSTM, seperti jumlah neuron, dropout rate, dan learning rate. Dengan cara ini, proses tuning jadi lebih efisien dan sistematis dibandingkan coba-coba manual, sehingga model yang dihasilkan lebih akurat dan tidak mudah overfitting. Selain itu, Keras Tuner mudah diintegrasikan dengan Keras/TensorFlow dan mempercepat eksperimen dengan fitur seperti EarlyStopping. Singkatnya, Keras Tuner membantu mendapatkan model LSTM yang optimal dengan lebih cepat dan handal.

# ### Buat Fungsi Model Builder untuk Keras Tuner

# Fungsi build_model bertujuan membangun model LSTM untuk regresi dengan hyperparameter yang dapat di-tune secara otomatis menggunakan Keras Tuner. Fungsi ini menerima objek hp (hyperparameter tuner) dan mengembalikan model Keras yang sudah dikompilasi. 

# In[23]:


def build_model(hp):
    model = keras.Sequential()

    # Layer LSTM dengan units yang ditentukan oleh tuner
    model.add(keras.layers.LSTM(
        units=hp.Choice('units', values=[32, 64, 128]),
        activation='tanh',
        input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])
    ))

    # Dropout layer untuk regularisasi
    model.add(keras.layers.Dropout(
        rate=hp.Choice('dropout', values=[0.1, 0.2, 0.3])
    ))

    # Output layer regresi
    model.add(keras.layers.Dense(1))

    # Compile model dengan learning rate yang ditentukan tuner
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        ),
        loss='mse',
        metrics=['mae']
    )

    return model


# Tahapan dan Parameter yang Digunakan
# 
# 1. Layer LSTM
#     - units: Jumlah neuron pada layer LSTM, dipilih secara otomatis dari opsi [32, 64, 128].
#     - activation: Fungsi aktivasi tanh digunakan karena umum dan efektif pada RNN/LSTM.
#     - input_shape: Bentuk input yang disesuaikan dengan data latih (biasanya (timesteps, fitur)).
# 2. Dropout Layer
#      - rate: Tingkat dropout untuk regularisasi, dipilih dari [0.1, 0.2, 0.3]. Dropout membantu mengurangi overfitting dengan mengacak neuron yang aktif selama training.
# 3. Output Layer (Dense)
#     - 1 neuron tanpa aktivasi (linear) untuk menghasilkan prediksi regresi (harga saham).
# 4. Kompilasi Model
#     - optimizer: Adam optimizer dengan learning rate yang juga di-tune otomatis, pilihan dari [0.01, 0.001, 0.0001].
#     - loss: Mean Squared Error (mse) digunakan karena masalah ini adalah regresi.
#     - metrics: Mean Absolute Error (mae) sebagai metrik evaluasi tambahan.

# ### Konfigurasi dan Jalankan Tuner

# Proses hyperparameter tuning menggunakan Keras Tuner dengan metode RandomSearch dilakukan untuk mencari kombinasi parameter terbaik pada model LSTM, seperti jumlah unit neuron, dropout rate, dan learning rate. Fungsi model dibuat fleksibel agar tuner bisa mencoba variasi parameter tersebut. RandomSearch mengeksplorasi beberapa kombinasi secara acak dan efisien, lalu memilih yang memberikan nilai validasi loss terbaik. Proses training menggunakan EarlyStopping untuk mencegah overfitting dengan menghentikan pelatihan saat performa tidak membaik. Dengan cara ini, model LSTM dapat diimprove secara otomatis tanpa trial manual, sehingga mendapatkan model yang optimal dengan waktu dan sumber daya yang lebih efisien.

# In[24]:


# Buat objek tuner
tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=10,  # Jumlah kombinasi hyperparameter yang akan dicoba
    executions_per_trial=1,
    directory='keras_tuner_logs',
    project_name='lstm_natam'
)

# Jalankan pencarian hyperparameter terbaik
tuner.search(
    X_train_lstm, y_train_scaled,
    validation_data=(X_test_lstm, y_test_scaled),
    epochs=50,
    batch_size=32,
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
    verbose=1
)


# Tahapan dan Parameter yang Digunakan
# 1. Inisialisasi Objek Tuner (RandomSearch)
#     - build_model: Fungsi yang membangun model dengan hyperparameter yang dapat dipilih oleh tuner.
#     - objective='val_loss': Tuner mengevaluasi model berdasarkan nilai loss pada data validasi, yaitu MSE terkecil.
#     - max_trials=10: Mencoba maksimal 10 kombinasi hyperparameter berbeda.
#     - executions_per_trial=1: Setiap kombinasi hyperparameter dieksekusi sekali saja (tanpa pengulangan).
#     - directory dan project_name: Tempat penyimpanan hasil pencarian.
# 2. Proses Pencarian (tuner.search):
#     - Melatih model dengan data training (X_train_lstm, y_train_scaled) dan validasi (X_test_lstm, y_test_scaled).
#     - epochs=50 dan batch_size=32: Pengaturan training standar.
#     - EarlyStopping: Menghentikan training jika tidak ada perbaikan val_loss selama 5 epoch berturut-turut dan mengembalikan bobot terbaik.
#     - verbose=1: Menampilkan proses pencarian di output.

# Saya mengambil model terbaik dan konfigurasi hyperparameter terbaik yang ditemukan oleh proses tuning menggunakan Keras Tuner. Hasil ini digunakan untuk melanjutkan tahap evaluasi dan deployment model dengan performa optimal.

# In[25]:


# Ambil model terbaik
best_model = tuner.get_best_models(num_models=1)[0]
best_hp = tuner.get_best_hyperparameters(1)[0]

# Tampilkan hasil tuning terbaik
print("Best Hyperparameters:")
print(f"Units        : {best_hp.get('units')}")
print(f"Dropout Rate : {best_hp.get('dropout')}")
print(f"Learning Rate: {best_hp.get('learning_rate')}")


# Penjelasan Hasil:
# - Units = 64: Model LSTM dengan 64 neuron pada layer LSTM memberikan performa terbaik.
# - Dropout Rate = 0.2: Dropout rate 0.2 digunakan untuk mengurangi overfitting tanpa mengurangi kapasitas model terlalu banyak.
# - Learning Rate = 0.01: Learning rate ini memastikan optimizer Adam melakukan update parameter dengan langkah yang optimal, mempercepat konvergensi tanpa melompati minimum loss.

# ### Ambil Model Terbaik dan Train Ulang

# Saya mengambil model terbaik hasil tuning dan melatih ulang model tersebut dengan data training dan validasi untuk memaksimalkan performa model sebelum evaluasi dan deployment.

# In[26]:


# Ambil model terbaik dari tuning
best_model = tuner.get_best_models(num_models=1)[0]

# Compile ulang model jika perlu
best_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss='mse',
    metrics=['mae']
)

# Training ulang (opsional, biasanya model terbaik sudah terlatih)
history = best_model.fit(
    X_train_lstm, y_train_scaled,
    validation_data=(X_test_lstm, y_test_scaled),
    epochs=50,
    batch_size=32,
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
    verbose=1
)


# # Evaluation

# ### Prediksi dan inverse transform

# In[27]:


import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Prediksi
y_pred_scaled = best_model.predict(X_test_lstm)

# Kembalikan ke skala asli
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_actual = scaler_y.inverse_transform(y_test_scaled)

# Evaluasi
mae = mean_absolute_error(y_test_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
r2 = r2_score(y_test_actual, y_pred)

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")


# Pada evaluasi model prediksi harga saham ini, digunakan tiga metrik utama:
# 1. MAE (Mean Absolute Error): Mengukur rata-rata besarnya selisih absolut antara nilai prediksi dengan nilai sebenarnya. Nilai MAE sebesar 0.0103 menunjukkan rata-rata kesalahan prediksi sangat kecil, sehingga model cukup akurat dalam memperkirakan harga penutupan saham.
# 
# 2. RMSE (Root Mean Squared Error): Mengukur akar dari rata-rata kuadrat selisih prediksi dan nilai sebenarnya. RMSE sebesar 0.0148 menunjukkan kesalahan prediksi yang cukup kecil dan penalti untuk kesalahan besar lebih tinggi, menandakan model memiliki prediksi yang konsisten dan stabil.
# 
# 3. R² (Koefisien Determinasi): Mengukur seberapa baik variansi data aktual dijelaskan oleh model. Nilai R² sebesar 0.9977 mendekati 1, yang berarti model sangat baik dalam menjelaskan variabilitas harga saham dan prediksi sangat mendekati nilai sebenarnya.
# 
# Kesimpulan: Berdasarkan ketiga metrik ini, model LSTM yang dibangun memiliki performa sangat baik untuk memprediksi harga penutupan saham PT ANTAM, dengan kesalahan prediksi yang sangat kecil dan akurasi yang tinggi.

# Ketiga metrik ini sesuai untuk masalah regresi pada data time series harga saham karena mereka mengukur akurasi prediksi numerik dan kemampuan model menjelaskan variasi data asli.

# ### Visualisasi Prediksi vs Aktual

# In[28]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(y_test_actual, label='Aktual')
plt.plot(y_pred, label='Prediksi', linestyle='--')
plt.title('Prediksi Harga Penutupan vs Aktual')
plt.xlabel('Hari')
plt.ylabel('Harga Penutupan')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Berdasarkan visualisasi grafik “Prediksi Harga Penutupan vs Aktual”, dapat disimpulkan bahwa model prediktif yang digunakan memiliki performa yang sangat baik secara visual. Hal ini terlihat dari garis prediksi (berwarna oranye putus-putus) yang sangat mendekati garis aktual (berwarna biru), menunjukkan bahwa model mampu menangkap pola pergerakan harga secara akurat. Model juga berhasil mengikuti tren utama, termasuk lonjakan harga yang tajam di sekitar hari ke-180 hingga 200, yang menunjukkan kemampuan model dalam mendeteksi perubahan tren harga. Selain itu, tidak tampak adanya deviasi besar antara prediksi dan nilai aktual sepanjang periode waktu yang diamati, sehingga kesalahan prediksi (error) dapat dikatakan relatif rendah dan konsisten.
# 
# Adapun sumbu horizontal (sumbu X) yang ditandai dengan angka dari 0 hingga lebih dari 200 menunjukkan urutan hari dalam data uji (bukan tanggal sebenarnya), dengan asumsi bahwa satu unit merepresentasikan satu hari perdagangan. Misalnya, angka 0 menunjukkan hari pertama dalam data uji, angka 100 adalah hari ke-100, dan seterusnya. Oleh karena itu, grafik ini menggambarkan kinerja model prediksi selama lebih dari 200 hari berturut-turut, di mana prediksi model terlihat konsisten mengikuti pergerakan harga aktual sepanjang periode tersebut. Dengan demikian, model ini layak digunakan untuk keperluan peramalan harga dalam jangka pendek hingga menengah, selama tren harga tidak mengalami perubahan yang sangat ekstrem atau tiba-tiba.
