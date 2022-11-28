import os
import pandas as pd
import numpy as np
import streamlit as st
import globalVal
from util import data_clean

import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Gaze Tracking DashBoard")

# App title
st.title('Gaze Tracking DashBoard')

st.markdown('Wellcome to Gaze Tracking Dashboard!')

st.header('Data Collection Guide')


m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #3d94f6;
    color: #ffffff;
    height: 4em;
    width: 12em;
    border-radius:6px;
    border:1px solid #337fed;
    font-family:Arial;
    font-size:22px;
    font-weight: bold;
    margin: auto;
    display: block;
}

div.stButton > button:hover {
	background:linear-gradient(to bottom, #1e62d0 5%, #3d94f6 100%);
	background-color:#1e62d0;
}

div.stButton > button:active {
	position:relative;
	top:1px;
}

</style>""", unsafe_allow_html=True)  # used to customized button

dataCollector = st.button('Click to start data collection')
dataClean  = st.sidebar.checkbox('Click to start data clean')
if dataCollector:
    #os.system('cd /Users/2602651K/Documents/GitHub/Gaze_Research/dashboard/dataCollector/ && python collector.py')
    dataCollector = False
    st.success('Data collection is done')

if dataClean:
    data_clean(globalVal.dataCollector_path + 'test1.csv',
               globalVal.dataCollector_path + 'testfile_msg.tsv')
    st.success('Data clean is done')

fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = 420,
    domain = {'x': [0, 1], 'y': [0, 1]},
    #title = {'text': "Level of Concentration", 'font': {'size': 24}},
    gauge = {
        'axis': {'range': [None, 500], 'tickwidth': 1, 'tickcolor': "blue"},
        'bar': {'color': 'rgb(255,215,0)'},
        'bgcolor': "white",
        'borderwidth': 2,
        'bordercolor': "gray",
        'steps': [
            {'range': [0, 100], 'color': 'rgb(240,248,255)'},
            {'range': [100, 200], 'color': 'rgb(176,196,222)'},
            {'range': [200, 300], 'color': 'rgb(100,149,237)'},
            {'range': [300, 400], 'color': 'rgb(65,105,225)'},
            {'range': [400, 500], 'color': 'rgb(0,0,205)'}],
        'threshold': {
            'line': {'color': 'rgb(255,215,0)', 'width': 4},
            'thickness': 0.75,
            'value': 500}}))

fig.update_layout(paper_bgcolor = "white", font = {'color': 'rgb(85,146,239)', 'family': "Arial"})


col1, col2 = st.columns(2)
with col1:
    st.image(fig.to_image())