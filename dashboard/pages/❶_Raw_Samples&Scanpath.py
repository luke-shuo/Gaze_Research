import streamlit as st
import pandas as pd
import numpy as np
import globalVal
from plots import draw_raw
from util import fixation_detection   # util file to find fixation and saccade detection.
from util import saccade_detection
from plots import draw_scanpath
from plots import image_convert

gaze_data = globalVal.gaze_data
image_data = globalVal.image_data

@st.cache
def load_data():
    data = pd.read_csv(gaze_data)
    data.fillna(0, inplace=True)    # Using '0' to replace 'N\A'
    return data
data = load_data()

# Only left gaze data is used
x = np.array(data['x_cod_left_gaze'])     # x-coordinate
y = np.array(data['y_cod_left_gaze'])     # y-coordinate
time = np.array(data['time_stamp'])       # time stamp

# Fixation and saccade detection
Sfix, Efix = fixation_detection(x, y, time, missing=0.0, maxdist=15, mindur=50)
fixations = Efix

Ssac, Esac = saccade_detection(x, y, time, missing=0.0, minlen=50, maxvel=300, maxacc=400)
saccades = Esac

raw = draw_raw(x, y, dispsize = [1920, 1080], imagefile=image_data)
scanpath = draw_scanpath(fixations, saccades, dispsize = [1920, 1080], imagefile=image_data,
                        alpha=0.4)

raw = image_convert(raw)
scanpath = image_convert(scanpath)

col1, col2 = st.columns(2)
with col1:
    st.header("Raw Gaze Samples")
    st.image(raw)
with col2:
    st.header("Scan Path")
    st.image(scanpath)