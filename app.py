import streamlit as st
import numpy as np
import pickle
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import make_scorer, accuracy_score,precision_score
from sklearn.metrics import accuracy_score ,precision_score,recall_score,f1_score
from sklearn.model_selection import KFold,train_test_split,cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import model_selection


st.title('Aplikasi Data Mining - Klasifikasi Kredit Score Menggunakan Algoritma Naive Bayes Gaussian')
st.write("Dataset Yang digunakan adalah **Kredit Score** dari [Github saya](https://raw.githubusercontent.com/masaul/data-csv/main/credit_score)")
st.write("Nama: Muhammad Aulia Faqihuddin | NIM : 200411100027")
st.sidebar.header("Parameter Inputan")

kprYa = st.sidebar.text_input('KPR Aktif (0 OR 1)', 0)
kprTidak = st.sidebar.text_input('KPR Tidak Aktif (0 OR 1)', 1)
overdue31 = st.sidebar.text_input('Rata-rata Overdue 0 - 31 Hari (0 OR 1)', 1)
overdue45 = st.sidebar.text_input('Rata-rata Overdue 31 - 45 Hari (0 OR 1)', 0)
overdue60 = st.sidebar.text_input('Rata-rata Overdue 46 - 60 Hari (0 OR 1)', 1)
overdue90 = st.sidebar.text_input('Rata-rata Overdue 61 - 90 Hari (0 OR 1)', 0)
overdueUpto90 = st.sidebar.text_input('Rata-rata Overdue > 90 Hari (0 OR 1)', 0)
pendapatan = st.sidebar.text_input('Pendapatan Setahun Juta (0 - 1)', 0.582609)
durasiPinjaman = st.sidebar.text_input('Durasi Pinjaman Bulan (0 - 1)', 0.333333)
jumlahTanggungan = st.sidebar.text_input('Jumlah Tanggungan (0 - 1)', 0.333333)
data = {
    "KPR Aktif" : kprYa,
    "KPR Tidak Aktif" : kprTidak,
    "0 - 30 Days" : overdue31,
    "31 - 45 Days" : overdue45,
    "46 - 60 Days" : overdue60,
    "61 - 90 Days" : overdue90,
    "> 90 Days" : overdueUpto90,
    "Pendapatan Setahun Juta" : pendapatan,
    "Durasi Pinjaman Bulan" : durasiPinjaman,
    "Jumlah Tanggungan" : jumlahTanggungan
}
fitur = pd.DataFrame(data, index=[0])


creditScoreRaw = pd.read_csv("https://raw.githubusercontent.com/masaul/data-csv/main/credit_score.csv")
# dataCreditScore = creditScoreRaw.drop(columns=['risk_rating'])

st.subheader("Data Sebelum di Preprocessing")
st.write(creditScoreRaw)

dataCreditScore_withoutColumns= pd.DataFrame(creditScoreRaw, columns=['kode_kontrak','pendapatan_setahun_juta','durasi_pinjaman_bulan','jumlah_tanggungan','risk_rating'])

# Encode Data menjadi numerik
encodeAvarage = pd.get_dummies(creditScoreRaw['rata_rata_overdue'])
encodeKprAktif=pd.get_dummies(creditScoreRaw['kpr_aktif'])

# Menggabungkan data yang sudah di encode
concatCreditScoreRaw = pd.concat([dataCreditScore_withoutColumns, encodeKprAktif, encodeAvarage], axis=1)


dataframeRiskRating = pd.DataFrame(creditScoreRaw, columns=['risk_rating'])

# Menghapus class risk rating
dropRiskRating = concatCreditScoreRaw.drop(['risk_rating'], axis=1)

# Menggabungkan class
concatCreditScoreRaw2 = pd.concat([dropRiskRating, dataframeRiskRating], axis=1)


# Preprocessing
preprocessingData = pd.DataFrame(creditScoreRaw, columns=['pendapatan_setahun_juta','durasi_pinjaman_bulan','jumlah_tanggungan'])
preprocessingData.to_numpy()
scaler = MinMaxScaler()
resultPreprocessingData = scaler.fit_transform(preprocessingData)
resultPreprocessingData = pd.DataFrame(resultPreprocessingData, columns=['pendapatan_setahun_juta','durasi_pinjaman_bulan','jumlah_tanggungan'])

dropColumnPreprocessingData = concatCreditScoreRaw2.drop(['pendapatan_setahun_juta','durasi_pinjaman_bulan','jumlah_tanggungan'], axis=1)
concatCreditScoreRaw3 = pd.concat([dropColumnPreprocessingData, resultPreprocessingData], axis=1)

dropColumn_RiskRatingPre = concatCreditScoreRaw3.drop(['risk_rating'], axis=1)
resultData = pd.concat([dropColumn_RiskRatingPre, dataframeRiskRating], axis=1)

st.subheader("Data Setelah di Preprocessing")
st.write(resultData)


X = resultData.iloc[:,1:11].values
y = resultData.iloc[:,11].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
Y_pred = gaussian.predict(X_test) 
akurasi = accuracy_score(y_test,Y_pred)

st.subheader("Hasil Akurasi")
st.write("Didapatkan dari data training 70% dan Data Testing 30%")
st.write(akurasi)


# Parameter Inputan dari User
st.subheader('Parameter Inputan')
st.write(fitur)

# Load model
model = GaussianNB()
model.fit(X_train, y_train)

# save the model to disk
filename = 'finalized_model.pkl'
pickle.dump(model, open(filename, 'wb')) 

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))

prediksiModel = loaded_model.predict([[int(kprYa), int(kprTidak), int(overdue31), int(overdue45), int(overdue60), 
int(overdue90), int(overdueUpto90), float(pendapatan), float(durasiPinjaman), float(jumlahTanggungan)]])

st.subheader("Hasil prediksi model yang di inputkan")
st.code(prediksiModel)


result = loaded_model.score(X_test, y_test)
st.subheader("Akurasi dari model yang di inputkan")
st.write(result)


# apply the whole pipeline to data
# dataArray = [0, 1, 1, 0, 1, 0, 0, 0.582609, 0.333333, 0.333333]
# pred = loaded_model.predict([dataArray])
# st.write(pred)
        

