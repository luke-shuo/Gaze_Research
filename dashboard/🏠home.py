import os
import pickle
import pandas as pd
from psychopy import visual, monitors, gui, core
import numpy as np
import matplotlib.pyplot as plt
from titta import Titta, helpers_tobii as helpers
import streamlit as st
from dataCollector import collector

st.set_page_config(layout="wide",page_title="Gaze Tracking DashBoard")

# App title
st.title('Gaze Tracking DashBoard')
st.markdown("""
<style>
@font-face {
  font-family: 'Arial';
  font-style: normal;
  font-weight: 40;
  unicode-range: U+0000-00FF, U+0131, U+0152-0153, U+02BB-02BC, U+02C6, U+02DA, U+02DC, U+2000-206F, U+2074, U+20AC, U+2122, U+2191, U+2193, U+2212, U+2215, U+FEFF, U+FFFD;
}

    html, body, [class*="css"]  {
    font-family: 'Arial';
    font-size: 20px;
    }
    </style> """, unsafe_allow_html=True,)

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

</style>""", unsafe_allow_html=True)        #used to customized button

dataCollector = st.button('Click to start data collection')


if dataCollector:
    #os.system('cd /Users/2602651K/PycharmProjects/PsychoPy3/demos/ && python et_demo.py')
    collector.collect_data()
    dataCollector = False


