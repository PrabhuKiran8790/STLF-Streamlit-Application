import streamlit as st
import numpy as np
from pathlib import Path
import joblib as jbl
from predict import predict
from PIL import Image
import base64
from input_form import form

st.set_page_config(page_title='RBF NN Load', page_icon='⚡️', layout="wide", initial_sidebar_state="expanded", menu_items=None, )

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    return base64.b64encode(img_bytes).decode()

warangal_html = f'<img src="data:image/png;base64,{img_to_bytes("warangal.png")}" class="img-fluid" width="600" height="290">'

rbf_architecture = Image.open('blockdiagram.png')

def warangal(): return st.markdown(
    warangal_html, unsafe_allow_html=True,
)

col1, icol2, icol3 = st.columns([1,6,1])

with col1:
    st.write("")
with icol2:
    banner = st.image(Image.open('transco.jpeg'), use_column_width='auto', output_format='png')
with icol3:
    st.write("")

st.markdown("""
<div style="text-align:center"><h4>Electric Power Load Prediction on a 33/11 KV Substation
<p>(Godishala, Saidapur, Telangana, India)</p></div>
""", unsafe_allow_html=True)

st.write("# RBF Neural Network for Predicting Load")

choice = ['Predict']
param = st.sidebar.selectbox("Select any of the options below", choice)

if param == 'Predict':
    with st.sidebar:
            st.markdown("""
                        ---
                        **Inputs:**
                        - Temperature (T-1) (°C)  
                        - Temperature (T-24) (°C)  
                        - Humidity (T-1) (%)  
                        - Humidity (T-24) (%)  
                        - Season (Winter, Summer, Rainy)  
                        ---
                        **Outputs:**
                        - Temperature (°C)
                        - Humidity (%)
                        ---
                        """)

    form()
    
header_html = f'<img src="data:image/png;base64,{img_to_bytes("logos.png")}" class="img-fluid" width="300" height="170">'
st.sidebar.markdown(
    header_html, unsafe_allow_html=True,
)

st.sidebar.write("#\n#\n#")
st.sidebar.markdown("**Disclaimer:** This project is associated with the [Center for Artificial Intelligence and Deep Learning (CAIDL)](https://sru.edu.in/centers/caidl/) at [SR University](https://sru.edu.in).")