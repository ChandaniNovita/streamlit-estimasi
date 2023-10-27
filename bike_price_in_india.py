import pickle
import streamlit as st

model = pickle.load(open('bike_price_in_india.sav','rb'))

st.title('Estimasi Harga Sepeda di India')

model_year = st.number_input('Input Tahun Keluaran')
kms_driven = st.number_input('input Tempuh KM')
owner = st.number_input('Input Kepemilikan')
location = st.number_input('Input Lokasi')
mileage = st.number_input('Input Jarak Tempuh')
power = st.number_input('Input Kecepatan')
company = st.number_input('Input Perusahaan')
bike_model = st.number_input('Input Jenis Sepeda')
age = st.number_input('input Usia Sepeda')

predict = ''

if st.button('Estimasi Harga'):
    predict =model.predict(
        [[model_year,kms_driven,owner,location,mileage,power,company,bike_model,age]]
    )
    st.write ('Estimasi harga sepeda dalam Rupe :', predict)