import streamlit as st
import joblib
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


# display
# st.set_page_config(layout='wide')
st.set_page_config(page_title="20-154 & 20-046", page_icon='icon.png')


@st.cache_resource()
def progress():
    with st.spinner('Wait for it...'):
        time.sleep(5)


st.title("UAS PENDAT")

dataframe, preporcessing, modeling, implementation = st.tabs(
    ["Data Finance", "Prepocessing", "Modeling", "Implementation"])

    #          'Overweight', 'Obesity', 'Extreme Obesity']
# label = ['Extremely Weak', 'Weak', 'Normal',
with dataframe:
    progress()
    url = "https://raw.githubusercontent.com/aliefakbrp/dataset/main/INDF.JK.csv"
    st.markdown(
        f'[Dataset Indofood]({url})')
    st.write('Hasil Saham Indofood')

    dataset, ket = st.tabs(['Dataset', 'Ket Dataset'])
    with ket:
        st.write("""
                Column
                * Date
                * Open (Harga Pembukaan): Harga saham pada awal perdagangan pada hari tertentu. Ini merupakan harga pertama yang ditetapkan ketika pasar saham dibuka pada hari tersebut.
                * High (Harga Tertinggi): Harga saham tertinggi yang tercapai selama periode perdagangan pada hari tertentu. Ini mencerminkan titik harga tertinggi yang dicapai oleh saham tersebut sepanjang hari tersebut.
                * Low (Harga Terendah): Harga saham terendah yang tercapai selama periode perdagangan pada hari tertentu. Ini menunjukkan titik harga terendah yang dicapai oleh saham tersebut sepanjang hari tersebut.      
                * Close (Harga Penutupan): Harga saham pada akhir perdagangan pada hari tertentu. Ini merupakan harga terakhir yang ditetapkan ketika pasar saham ditutup pada hari tersebut.
                * Adj Close (Harga Penutupan Disesuaikan): Harga penutupan yang disesuaikan untuk memperhitungkan pembagian saham, dividen, atau perubahan lainnya yang dapat mempengaruhi harga saham. Harga penutupan yang disesuaikan memberikan gambaran yang lebih akurat tentang kinerja investasi seiring waktu.
                * Volume: Jumlah saham yang diperdagangkan selama periode perdagangan pada hari tertentu. Volume mencerminkan seberapa besar minat dan aktivitas perdagangan yang terjadi pada saham tersebut. Volume yang tinggi menunjukkan minat yang kuat, sedangkan volume yang rendah menunjukkan minat yang lebih sedikit.
                
                """)
    with dataset:
        dt = pd.read_csv(
            'https://raw.githubusercontent.com/aliefakbrp/dataset/main/INDF.JK.csv')
      #   dt = dt.drop('Unnamed: 0', axis=1)
        st.dataframe(dt)


with preporcessing:
    progress()
    minmax, std = st.tabs(['Minmax', 'Standart Scaler'])
    with minmax:
      dtminmax = pd.read_csv('https://raw.githubusercontent.com/aliefakbrp/dataset/main/univaried_trans_3fitur.csv')
      dfminmax = pd.get_dummies(dtminmax)
      st.dataframe(dfminmax)
    with std:
      dtstd = pd.read_csv('https://raw.githubusercontent.com/aliefakbrp/dataset/main/univaried_trans_3fitur.csv')
      dfstd = pd.get_dummies(dtstd) 
      # value = dfstd.drop('Unnamed: 0', axis=1)
      st.dataframe(dfstd)
#     st.write(value) 

#     st.write(value)

#     time.sleep(2)
#     dtminmax = pd.read_csv('https://raw.githubusercontent.com/aliefakbrp/dataset/main/univaried_trans_3fitur.csv')
#     dfminmax = pd.get_dummies(dtminmax) 
#     dfminmax = dfminmax.drop('Unnamed: 0', axis=1)
#     st.dataframe(dfminmax)

#     value = a.radio("gender", options, 1)



with modeling:
    progress()
    # pisahkan fitur dan label
    dtminmax = pd.read_csv('https://raw.githubusercontent.com/aliefakbrp/dataset/main/univaried_trans_3fitur.csv')
    df = pd.get_dummies(dtminmax)
    df = df.drop('Unnamed: 0', axis=1)
    X = df.drop('Xt', axis=1)
    y = df['Xt']
    # split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=1)
    knc, regress = st.tabs(
        ["KNeighborsClassifier", "Regressor"])

    with knc:
        progress()
        clf = joblib.load('model_knn_pkl.pkl')
        y_test=y_test.drop("")
        st.write(y_test)
        y_pred_clf = clf.predict(X_test)
        st.write(y_pred_clf)
        akurasi_clf = accuracy_score(y_test, y_pred_clf)
        label_clf = pd.DataFrame(
            data={'Label Test': y_test, 'Label Predict': y_pred_clf}).reset_index()
        st.success(f'akurasi terhadap data test = {akurasi_clf}')
        st.dataframe(label_clf)
    with knc:
        progress()
        knn = joblib.load('model_knn_pkl.pkl')

        y_pred_knn = knn.predict(X_test)
        akurasi_knn = accuracy_score(y_test, y_pred_knn)
        label_knn = pd.DataFrame(
            data={'Label Test': y_test, 'Label Predict': y_pred_knn}).reset_index()
        st.success(f'akurasi terhadap data test = {akurasi_knn}')
        st.dataframe(label_knn)
    with dtc:
        progress()
        d3 = joblib.load('d3.pkl')
        y_pred_d3 = d3.predict(X_test)
        akurasi_d3 = accuracy_score(y_test, y_pred_d3)
        label_d3 = pd.DataFrame(
            data={'Label Test': y_test, 'Label Predict': y_pred_d3}).reset_index()
        st.success(f'akurasi terhadap data test = {akurasi_d3}')
        st.dataframe(label_d3)

with implementation:
    # height
    height = st.number_input('Tinggi', value=174)
    # weight
    weight = st.number_input('Berat', value=96)
    # gender
    gander = st.selectbox('Jenis Kelamin', ['Laki-Laki', 'Prempuan'])
    gander_female = 1 if gander == 'Prempuan' else 0
    gander_male = 1 if gander == 'Laki-Laki' else 0

    data = np.array([[height, weight, gander_female, gander_male]])
    model = st.selectbox('Pilih Model', ['MLP', 'KNN', 'D3'])
    if model == 'MLP':
        y_imp = clf.predict(data)
    elif model == 'KNN':
        y_imp = knn.predict(data)
    else:
        y_imp = d3.predict(data)
    st.success(f'Model yang dipilih = {model}')
    st.success(f'Data Predict = {label[y_imp[0]]}')