import streamlit as st
import pandas as pd
import numpy as np
import cv2
import os
import shutil
import globalVal
from plots import draw_fixations
from plots import draw_heatmap
from plots import image_convert
from plots import draw_arrows
from plots import draw_boundingbox
from util import fixation_detection   # util file to find fixation and saccade detection.
from util import fix_count
from util import video_generator
from util import fixations_integration
from util import step_count
from util import turnback_count
from plots import draw_guage

# get data address from globalVal.py
imageFileName = []
for filename in os.listdir(globalVal.images_path):
    imageFileName.append(filename[-8:-5])
image_name = st.sidebar.selectbox('Select the image', imageFileName)
image_csv = pd.read_csv(globalVal.dataset_path+'image.csv')
image_list = np.array(image_csv['0']).tolist()
dataset_index = image_list.index(image_name)

dataset = globalVal.dataset_path + 'dataset%d.csv' % dataset_index
image = globalVal.images_path + image_name + '.jpeg'

fixation_addr = globalVal.fixations_path
bounding_image = globalVal.bounding_image_path
video_addr = globalVal.video_output_path
aoi_loc = []
stimulus_duration = 60

# check whether fixation_addr exists.
if not os.path.exists(fixation_addr):
    os.mkdir(fixation_addr)
else:
    shutil.rmtree(fixation_addr)
    os.mkdir(fixation_addr)

# Load gaze data from dataset.
@st.cache
def load_data():
    data = pd.read_csv(dataset)
    data.fillna(0, inplace=True)    # Using '0' to replace 'N\A'
    return data
data = load_data()

# Only left gaze data is used
x = np.array(data['x_cod_left_gaze'])     # x-coordinate
y = np.array(data['y_cod_left_gaze'])     # y-coordinate
time = np.array(data['time_stamp'])       # time stamp

# integration index
integration_index = st.sidebar.slider('Integration index', min_value=0.1, max_value=2.0, value=1.0)
circle_size = 450-st.sidebar.slider('Choose the circle size', 50, 400,300)
with st.sidebar.form(key='my_form'):
    submit_button = st.form_submit_button(label='Start Generation')

# Fixation and saccade detection
Sfix, Efix = fixation_detection(x, y, time, missing=0.0, maxdist=15, mindur=50)
fixations = Efix
fixations_inte = fixations_integration(time=time, fixations=fixations, dur=integration_index)

fixation = draw_fixations(fixations, dispsize = [1920, 1080], imagefile=image,
                          alpha=0.4, size=circle_size, savefilename=bounding_image)
fixation_inte = draw_fixations(fixations_inte, dispsize = [1920, 1080], imagefile=image,
                               alpha=0.4)

fixation = image_convert(fixation)
fixation_inte = image_convert(fixation_inte)

# generate default transition video
if not os.path.exists(video_addr + image_name +'_default' + '.mp4'):
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H264
    videowrite = cv2.VideoWriter(video_addr + image_name+'_default' + '.mp4', fourcc, len(fixations)/stimulus_duration,
                                     (1920,1080))
    for j in range(len(fixations)):
        fixation_slice = draw_fixations([fixations[j]], dispsize=[1920, 1080],
                                        savefilename=fixation_addr+'/1.jpg',
                                        imagefile=image, alpha=0.4)
        draw_arrows(fixations=fixations, index=j, imagefile=fixation_addr+'/1.jpg')
        img = cv2.imread(fixation_addr + '/1.jpg')
        videowrite.write(img)
    cv2.destroyAllWindows()
    videowrite.release()

# generate customized transition video
if submit_button:
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H264
    videowrite = cv2.VideoWriter(video_addr + 'fixation_output' + '.mp4', fourcc, 1/integration_index,
                                 (1920, 1080))
    for j in range(len(fixations_inte)):
        fixation_slice = draw_fixations([fixations_inte[j]], dispsize=[1920, 1080],
                                        savefilename=fixation_addr + '/1.jpg',
                                        imagefile=image, alpha=0.4)
        draw_arrows(fixations=fixations_inte, index=j, imagefile=fixation_addr + '/1.jpg')
        img = cv2.imread(fixation_addr + '/1.jpg')
        videowrite.write(img)
    cv2.destroyAllWindows()
    videowrite.release()

# three line chart based on fixation steps
step_list = step_count(fixations)
fix_dur, turnback_count, turnback_index = turnback_count(fixations)

# heatmap results
heatmap = draw_heatmap(fixations, dispsize = [1920, 1080], imagefile=image,
                       alpha=0.4)
heatmap  = image_convert(heatmap)

guage = draw_guage(368)

col1, col2 = st.columns(2)
with col1:
    st.header('Concentration Level')
    st.image(guage)
with col2:
    st.header("Fixations")
    st.image(fixation)

st.header("Top-3 Concentrated Areas")
col5, col6, col7 = st.columns(3)
with col5:
    st.image('/Users/lukeshuo/Documents/GitHub/Gaze_Research/dashboard/im4.jpeg')
with col6:
    st.image('/Users/lukeshuo/Documents/GitHub/Gaze_Research/dashboard/im5.jpeg')
with col7:
    st.image('/Users/lukeshuo/Documents/GitHub/Gaze_Research/dashboard/im6.jpeg')

if submit_button:
    st.header("Transition Video")
    st.video(data=video_addr + 'fixation_output.mp4')
else:
    st.header("Transition Video")
    st.video(data=video_addr + image_name+'_default.mp4')

st.header("Heatmap")
st.image(heatmap)

