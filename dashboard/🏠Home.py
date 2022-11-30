import os
import pandas as pd
import numpy as np
import streamlit as st
import globalVal
from util import data_clean

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
