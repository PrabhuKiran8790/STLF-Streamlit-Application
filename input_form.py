import streamlit as st
import numpy as np
from predict import predict
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def form():
    ae_model = tf.keras.models.load_model('Autoencoder_ls_10_bs_64.h5')
    with st.form(key='load'):
        icol1, icol2 = st.columns(2)
        
        with icol1:
            st.write("##### Please enter the following inputs for Temperature and Humidity Prediction:")
            col1, col2 = st.columns(2)
            with col1:
                T1 = st.number_input("Temperature (T-1) (°C)", min_value=0.0, max_value=100.0, value=32.0, step=0.1)
                H1 = st.number_input("Humidity (T-1) (%)", min_value=0.0, max_value=100.0, value=97.0, step=0.1)
                T1 = (T1 * 9/5) + 32
            with col2:
                T2 = st.number_input("Temperature (T-2) (°C)", min_value=0.0, max_value=50.0, value=33.0, step=0.1)
                T2 = (T2 * 9/5) + 32
                H2 = st.number_input("Humidity (T-2) (%)", min_value=0.0, max_value=100.0, value=98.0, step=0.1)
            season = st.selectbox("Season", ["Winter", "Summer", "Rainy"])
            st.write("---")
            day = st.selectbox("Day", ["weekday", "weekend"])
            if day == "weekday":
                day = 0
            if day == "weekend":
                day = 1
            if season == "Winter":
                season = 0
            elif season == "Summer":
                season = 1
            else:
                season = 2 # rainy
        with icol2:
            st.write("##### Please enter the following inputs for Load Prediction:")
            col1, col2 = st.columns(2)
            with col1:
                pt1 = st.number_input('Load (T-1)', min_value=0, max_value=8000, value=1500, step=100)
                pt3 = st.number_input('Load (T-3)', min_value=0, max_value=8000, value=1200, step=100)
                pt24 = st.number_input('Load (T-24)', min_value=0, max_value=8000, value=1000, step=100)
                pt72 = st.number_input('Load (T-72)', min_value=0, max_value=8000, value=800, step=100)
            with col2:
                pt2 = st.number_input('Load (T-2)', min_value=0, max_value=8000, value=1300, step=100)
                pt4 = st.number_input('Load (T-4)', min_value=0, max_value=8000, value=1000, step=100)
                pt48 = st.number_input('Load (T-48)', min_value=0, max_value=8000, value=800, step=100)
                pt96 = st.number_input('Load (T-96)', min_value=0, max_value=8000, value=600, step=100)
        
        if submit := st.form_submit_button("Submit"):
            temp_humidity_inputs = np.array([T1, T2, H1, H2, season])
            temp = predict(temp_humidity_inputs, 'models/temp_optimal_info.jbl')
            humidity = predict(temp_humidity_inputs, 'models/humidity_optimal_info.jbl')
            st.write(temp_humidity_inputs)
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"#### Predicted Temperature: {((temp[0][0] - 32)*5)/9:.1f} °C")
                st.write(f"#### Predicted Humidity: {humidity[0][0]:.1f} %")
            with col2:
                inputs = np.array([[pt1, pt2, pt3, pt4, pt24, pt48, pt72, pt96, day, season, temp[0][0], humidity[0][0]]])
                load = predict(inputs, 'models/load_optimal_info.jbl', load=True, ae_filepath='Autoencoder_ls_10_bs_64.h5')
                st.write(f"#### Predicted Load: {load[0][0]}")
                