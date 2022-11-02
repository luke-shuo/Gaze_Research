import streamlit as st
import pandas as pd
import numpy as np
import os
import globalVal
import shutil
from util import fixation_detection   # util file to find fixation and saccade detection.
from util import saccade_detection
from util import fixations_slice
from util import video_generator
from util import fixations_integration
from plots import draw_heatmap
from plots import image_convert

gaze_data = globalVal.gaze_data
image_data = globalVal.image_data
heatmap_addr = globalVal.heatmap_addr
video_addr = globalVal.video_data

if not os.path.exists(heatmap_addr):
    os.mkdir(heatmap_addr)
else:
    shutil.rmtree(heatmap_addr)
    os.mkdir(heatmap_addr)

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


integration_index = st.sidebar.number_input('Integration index', 0.5)

# Fixation and saccade detection
Sfix, Efix = fixation_detection(x, y, time, missing=0.0, maxdist=15, mindur=50)
fixations = Efix
fixations_inte = fixations_integration(time=time, fixations=fixations, dur=integration_index)


Ssac, Esac = saccade_detection(x, y, time, missing=0.0, minlen=50, maxvel=300, maxacc=400)
saccades = Esac

timeline = st.sidebar.slider('Timeline of fixations', 0, len(fixations_inte)-1)

heatmap = draw_heatmap(fixations, dispsize = [1920, 1080], imagefile=image_data,
                        alpha=0.4)

heatmap  = image_convert(heatmap)

bs = list()
for j in range(len(fixations_inte)):
    heatmap_slice = draw_heatmap(fixations_inte[j:j+1], dispsize = [1920, 1080],savefilename=heatmap_addr+'/%d.jpg' %j, imagefile=image_data,
                        alpha=0.4)
    heatmap_slice = image_convert(heatmap_slice)
    bs.append(heatmap_slice[np.newaxis, :])
fig_list = np.concatenate(bs, axis=0)   #fig_list contaning a list of figures, which are heatmap in different time

#generate heatmap video
video_generator(outname='heatmap_output', slice=fixations_inte, input_addr=heatmap_addr, fps=1/integration_index, size=(1920,1080))

st.header("Heatmap")
st.image(heatmap)

col1, col2 = st.columns(2)
with col1:
    st.header("Heatmap in selected time")
    st.image(fig_list[timeline])
with col2:
    st.header("Transition Video")
    st.video(data=video_addr+'heatmap_output.mp4')