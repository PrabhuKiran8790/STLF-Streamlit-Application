import streamlit as st
import numpy as np
from predict import predict
import tensorflow as tf

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
            temp = predict(temp_humidity_inputs, "temperature_Metadata_N_12_P_11_bs_32.jbl", "temperature_RBF_ANN_model_bs_32_N_12_P_11.h5")
            temp = (108 - 50) * temp + 50
            humidity = predict(temp_humidity_inputs, "humidity_Metadata_N_12_P_10_bs_128.jbl", "humidity_RBF_ANN_model_bs_128_N_12_P_10.h5")
            humidity = (102 - 14) * humidity + 14
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"#### Predicted Temperature: {((temp[0][0] - 32)*5)/9:.1f} °C")
                st.write(f"#### Predicted Humidity: {humidity[0][0]:.1f} %")
            with col2:
                ae_inputs = [pt1, pt2, pt3, pt4, pt24, pt48, pt72, pt96, day, season, temp[0][0], humidity[0][0]]
                ae_outputs = ae_model.predict([ae_inputs])
                ae_out = list(ae_outputs[0])
                ae_out.pop(3)
                load = predict(np.array(ae_out), "load_Metadata_N_19_P_12_bs_32.jbl", "load_RBF_ANN_model_bs_32_N_19_P_12.h5")
                load = (np.array([[6619.96585773]]) - np.array([458.02005145])) * load + np.array([458.02005145])
                st.write(f"#### Predicted Load: {load[0][0]:.1f} kW")