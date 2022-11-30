import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from collections import OrderedDict
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.svm import SVC
import plotly.express as px
from sklearn.utils.validation import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import altair as alt
from PIL import Image
image = Image.open('tekanan.png')
img = Image.open('kol.jpg')
rasio = Image.open('rasio.png')

#intial template
px.defaults.template = "plotly_dark"
px.defaults.color_continuous_scale = "reds"

df = pd.read_csv("https://raw.githubusercontent.com/febyfadlilah/dataset/main/drug200.csv")

st.title("PREDIKSI JENIS OBAT YANG RELEVAN")
st.caption("FEBI FADLILAH NUR AMINAH | 200411100115")
st.write("Aplikasi ini merupakan aplikasi yang digunakan untuk memprediksi jenis obat yang cocok berdasarkan kualifikasi yang di inputkan.")
home,preprocessing, modeling, implementation = st.tabs(["Home", "Preprocessing", "Modeling", "Implementation"])

with home:
    st.header("About DataSet")
    st.write("Pada aplikasi ini dataset yang digunakan adalah dataset yang berisikan klasifikasi jenis obat.")
    st.write("Dataset yang digunakan pada aplikasi ini diambil dari https://www.kaggle.com/datasets/prathamtripathi/drug-classification")
    st.write("Sasaran dari klasifikasi ini adalah Jenis Obat. Untuk Menentukan Jenis Obat tersebut maka diperlukan pendukung seperti : ")
    st.write("* Usia")
    st.write("* Jenis Kelamin")
    st.write("* Tingkat Tekanan Darah")
    st.write("Tekanan darah menunjukkan seberapa kuat jantung memompa darah ke seluruh tubuh Anda. Ukuran ini merupakan salah satu tanda vital tubuh yang sering dijadikan acuan untuk melihat kesehatan tubuh secara umum dan harus dipantau secara berkala.")
    st.write("Tekanan darah setiap orang berbeda-beda karena berbagai macam faktor. Salah satunya adalah usia. Semakin bertambah usia seseorang, semakin tinggi pula kisaran normal tekanan darahnya. Melalui artikel ini, Anda akan mengetahui batas tekanan darah normal berdasarkan usia.")
    st.write("Jika Tekanan darah anda kurang dari Normal maka tergolong rendah, dan jika lebih dari Normal maka tergolong tinggi")
    st.write("Tabel Tekanan Darah dapat dilihat dibawah ini : ")
    st.image(image, caption='Tabel Tekanan Darah')
    st.write("* Tingkat Kolesterol")
    st.write("Kolesterol merupakan senyawa lemak yang diproduksi oleh berbagai sel dalam tubuh, dan sekitar seperempat kolesterol yang dihasilkan dalam tubuh diproduksi oleh sel-sel hati. Pada dasarnya tubuh membutuhkan kolesterol untuk tetap sehat. Namun bila berlebih dapat berbahaya.")
    st.write("Tabel Tingkat Kolesterol dapat dilihat dibawah ini : ")
    st.image(img, caption='Tabel Tingkat Kolesterol')
    st.write("* Rasio Na terhadap Kalium")
    st.write("Menurut penelitian, dari 19 subyek yang memiliki rasio asupan natrium:kalium baik, 15 (78,95%) subyek tidak mengalami hipertensi atau memiliki tekanan darah diastolik normal, sedangkan 50% subyek memiliki tekanan darah sistolik tidak normal, seperti yang tersaji pada tabel berikut")
    st.image(rasio, caption='Tabel Rasio Na:K dengan Hipertensi')
    st.write("Penggunaan beberapa jenis obat secara umum adalah Obat X dan Y adalah obat untuk rasio kalium dalam darah. Obat C adalah untuk orang dengan riwayat tekanan darah rendah. Obat A dan B untuk tekanan darah tinggi, dengan catatan usia dibawah 50 tahun meminum obat jenis A dan yang diatas 50 tahun meminum obat jenis B.")
    st.write("Namun pada aplikasi ini, pengguna hanya perlu menginputkan kualifikasi sesuai kondisi tubuh masing-masing untuk mengetahui jenis obat mana yang cocok.")
with preprocessing:
    st.write("""# Preprocessing Data""")
    st.write("Notes : Masukkan dataset yang ada pada link berikut https://www.kaggle.com/datasets/prathamtripathi/drug-classification")
    uploaded_files = st.file_uploader("Upload file CSV", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file)
    show = st.button("Show Data")
    if show :
        st.dataframe(df)
    st.subheader("Berikut ini adalah pilihan untuk preprocessing Data")
    st.write("Transformasi Fitur Kategorikal to Numerik")
    jk = st.checkbox('Jenis Kelamin')
    td = st.checkbox('Tekanan Darah')
    kl = st.checkbox('Tingkat Kolesterol')
    submit = st.button("Transformasi")
    jk_trans = pd.get_dummies(df[["Sex"]])
    td_trans = pd.get_dummies(df[["BP"]])
    kl_trans = pd.get_dummies(df[["Cholesterol"]])
    transformasi = pd.concat([jk_trans,td_trans,kl_trans], axis=1)
    dataHasil = pd.concat([df,transformasi], axis = 1)
    X = dataHasil.drop(columns=["Drug","Sex","BP","Cholesterol"])
    y = dataHasil.Drug
    if jk and td and kl :
        if submit :
            st.dataframe(transformasi)
            # dataHasil = pd.concat([df,transformasi], axis = 1)
            # X = dataHasil.drop(columns=["Drug","Sex","BP","Cholesterol"])
            # y = dataHasil.Drug
            st.write("## Hasil Normalisasi Data Dengan Data Numerik Semua")
            st.dataframe(X)
    else :
        if submit :
            st.error('Kamu harus memilih semua Fitur untuk bisa lanjut ke proses selanjutnya', icon="🚨")
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = le.transform(y)
    le.inverse_transform(y)
    labels = pd.get_dummies(dataHasil.Drug).columns.values.tolist()
    agree = st.checkbox('Klik Checkbox disamping untuk lanjut ke Proses Normalisasi Min Max Scaler')
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)
    # X_train.shape, X_test.shape, y_train.shape, y_test.shape
    tombol = st.checkbox('Klik Checkbox disamping untuk lanjut ke Normalisasi StandardScaler data Training dan data Testing')
    ok = st.button("Normalisasi")
    if agree :
        if jk and td and kl:
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            scaler.fit(X)
            X = scaler.transform(X)
            # st.write("""# Normalisasi Min Max Scaler""")
            # st.dataframe(X)
        else :
            st.error('Kamu harus memilih semua Fitur untuk di Transformasi agar bisa lanjut ke proses Normalisasi', icon="🚨")
    
    if tombol : 
        if jk and td and kl:
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
            # st.write("""# Normalisasi Standard Scaler Data Training""")
            # st.dataframe(X_train)
            # st.write("""# Normalisasi Standard Scaler Data Training""")
            # st.dataframe(X_test)
        else :
            st.error('Kamu harus memilih semua Fitur untuk di Transformasi agar bisa lanjut ke proses Normalisasi', icon="🚨")
    if ok :
        if agree :
            st.write("""# Normalisasi Min Max Scaler""")
            st.dataframe(X)
    if ok :
        if tombol :
            st.write("""# Normalisasi Standard Scaler Data Training""")
            st.dataframe(X_train)
            st.write("""# Normalisasi Standard Scaler Data Training""")
            st.dataframe(X_test)
    


with modeling:
    st.write("""# Modeling """)
    st.subheader("Berikut ini adalah pilihan untuk Modeling")
    st.write("Pilih Model yang Anda inginkan untuk Cek Akurasi")
    naive = st.checkbox('Naive Bayes')
    kn = st.checkbox('K-Nearest Neighbor')
    des = st.checkbox('Decision Tree')
    mod = st.button("Modeling")

    # NB
    GaussianNB(priors=None)

    # Fitting Naive Bayes Classification to the Training set with linear kernel
    nvklasifikasi = GaussianNB()
    nvklasifikasi = nvklasifikasi.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = nvklasifikasi.predict(X_test)
    
    y_compare = np.vstack((y_test,y_pred)).T
    nvklasifikasi.predict_proba(X_test)
    akurasi = round(100 * accuracy_score(y_test, y_pred))

    # KNN 
    K=10
    knn=KNeighborsClassifier(n_neighbors=K)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)

    skor_akurasi = round(100 * accuracy_score(y_test,y_pred))

    # DT

    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    # prediction
    dt.score(X_test, y_test)
    y_pred = dt.predict(X_test)
    #Accuracy
    akurasiii = round(100 * accuracy_score(y_test,y_pred))

    if naive :
        if mod :
            st.success('Model Naive Bayes accuracy score: {0:0.2f}'. format(akurasi))
    if kn :
        if mod:
            st.success("Model KNN accuracy score : {0:0.2f}" . format(skor_akurasi))
    if des :
        if mod :
            st.success("Model Decision Tree accuracy score : {0:0.2f}" . format(akurasiii))
    
    eval = st.button("Evaluasi semua model")
    if eval :
        # st.snow()
        source = pd.DataFrame({
            'Nilai Akurasi' : [akurasi,skor_akurasi,akurasiii],
            'Nama Model' : ['Naive Bayes','KNN','Decision Tree']
        })

        bar_chart = alt.Chart(source).mark_bar().encode(
            y = 'Nilai Akurasi',
            x = 'Nama Model'
        )

        st.altair_chart(bar_chart,use_container_width=True)


with implementation:
    st.write("# Implementation")
    Age = st.number_input('Masukkan Umur ')
    # sex = st.number_input('Masukkan Jenis Kelamin')
    Sex = st.radio(
    "Masukkan Jenis Kelamin Anda",
    ('Laki-laki','Perempuan'))
    if Sex == "Laki-laki":
        Sex_Female = 0
        Sex_Male = 1
    elif Sex == "Perempuan" :
        Sex_Female = 1
        Sex_Male = 0

    BP = st.radio(
    "Masukkan Tekanan Darah Anda",
    ('Tinggi','Normal','Rendah'))
    if BP == "Tinggi":
        BP_High = 1
        BP_LOW = 0
        BP_NORMAL = 0
    elif BP == "Normal" :
        BP_High = 0
        BP_LOW = 0
        BP_NORMAL = 1
    elif BP == "Rendah" :
        BP_High = 0
        BP_LOW = 1
        BP_NORMAL = 0

    Cholesterol = st.radio(
    "Masukkan Kadar Kolestrol Anda",
    ('Tinggi','Normal'))
    if Cholesterol == "Tinggi" :
        Cholestrol_High = 1
        Cholestrol_Normal = 0 
    elif Cholesterol == "Normal":
        Cholestrol_High = 0
        Cholestrol_Normal = 1
        
    Na_to_K = st.number_input('Masukkan Rasio Natrium Ke Kalium dalam Darah')



    def submit():
        # input
        inputs = np.array([[Age, Sex_Female,Sex_Male, BP_High,BP_LOW,BP_NORMAL, Cholestrol_High,Cholestrol_Normal, Na_to_K]])
        # st.write(inputs)
        # baru = pd.DataFrame(inputs)
        # input = pd.get_dummies(baru)
        # st.write(input)
        # inputan = np.array(input)
        # import label encoder
        le = joblib.load("le.save")
        # scal = joblib.load("scaler.save")
        if akurasi > skor_akurasi and akurasi > akurasiii :
            model3 = joblib.load("nb.joblib")
        elif skor_akurasi > akurasi and skor_akurasi > akurasiii :
            model3 = joblib.load("knn.joblib")
        elif akurasiii > akurasi and akurasiii > skor_akurasi :
            model3 = joblib.load("tree.joblib")
        y_pred3 = model3.predict(inputs)
        st.success(f"Berdasarkan data yang Anda inputkan maka obat yang cocok untuk Anda adalah obat jenis : {le.inverse_transform(y_pred3)[0]}")

    all = st.button("Submit")
    if all :
        st.balloons()
        submit()
