import os
from data_collection import collector
import streamlit as st
import pandas as pd
import numpy as np
import globalVal
from util import fixation_detection   # util file to find fixation and saccade detection.
from util import saccade_detection
from plots import draw_raw
from plots import draw_fixations
from plots import draw_scanpath
from plots import draw_heatmap
from plots import image_convert

st.set_page_config(layout="wide")
# App title
st.title('Gaze Tracking DashBoard')
# data address
gaze_data = globalVal.gaze_data
image_data = globalVal.image_data

#Load data and using cache to speed up streamlit
@st.cache
def load_data():
    data = pd.read_csv(gaze_data)
    data.fillna(0, inplace=True)    # Using '0' to replace 'N\A'
    return data
data = load_data()

#uploaded_files = st.file_uploader("Upload image files", accept_multiple_files=True)
#for uploaded_file in uploaded_files:
#    print(type(uploaded_file))

# Only left gaze data is used
x = np.array(data['x_cod_left_gaze'])     # x-coordinate
y = np.array(data['y_cod_left_gaze'])     # y-coordinate
time = np.array(data['time_stamp'])       # time stamp

# Fixation and saccade detection
Sfix, Efix = fixation_detection(x, y, time, missing=0.0, maxdist=15, mindur=50)
fixations = Efix
Ssac, Esac = saccade_detection(x, y, time, missing=0.0, minlen=50, maxvel=300, maxacc=400)
saccades = Esac

# Draw raw, scanpath, heatmap, fixations
raw = draw_raw(x, y, dispsize = [1920, 1080], imagefile=image_data)

scanpath = draw_scanpath(fixations, saccades, dispsize = [1920, 1080], imagefile=image_data,
                        alpha=0.4)

heatmap = draw_heatmap(fixations, dispsize = [1920, 1080], imagefile=image_data,
                        alpha=0.4)

fixation = draw_fixations(fixations, dispsize = [1920, 1080], imagefile=image_data,
                        alpha=0.4)

# data collection
dataCollector = st.sidebar.button("Click to start collect data")
if dataCollector:
    #os.system('cd /Users/2602651K/PycharmProjects/PsychoPy3/demos/ && python et_demo.py')
    collector.collect_data()
    dataCollector = False

# Convert matplotlib image into numpy array
raw = image_convert(raw)
scanpath = image_convert(scanpath)
heatmap = image_convert(heatmap)
fixation = image_convert(fixation)


col1, col2 = st.columns(2)
with col1:
    st.header("Raw Gaze Samples")
    st.image(raw)

    st.header("Scan Path")
    st.image(scanpath)

with col2:
    st.header("Heatmap")
    st.image(heatmap)

    st.header("Fixations")
    st.image(fixation)

