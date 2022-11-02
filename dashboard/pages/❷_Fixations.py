import streamlit as st
import pandas as pd
import numpy as np
import cv2
import os
import shutil
import globalVal
from plots import draw_fixations
from plots import image_convert
from plots import draw_arrows
from plots import draw_boundingbox
from util import fixation_detection   # util file to find fixation and saccade detection.
from util import fix_count
from util import video_generator
from util import fixations_integration
from util import step_count
from util import turnback_count

# get data address from globalVal.py
gaze_data = globalVal.gaze_data
image_data = globalVal.image_data
fixation_addr = globalVal.fixation_addr
bounding_addr = globalVal.bounding_map_addr
video_addr = globalVal.video_data
aoi_loc = []

# check whether fixation_addr exists.
if not os.path.exists(fixation_addr):
    os.mkdir(fixation_addr)
else:
    shutil.rmtree(fixation_addr)
    os.mkdir(fixation_addr)

# Load gaze data from dataset.
@st.cache
def load_data():
    data = pd.read_csv(gaze_data)
    data.fillna(0, inplace=True)    # Using '0' to replace 'N\A'
    return data
data = load_data()

# This function is used to get aoi location by mouse click
def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    global aoi_loc
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        aoi_loc.append((x, y, 0))
        cv2.circle(img, (x, y), 2, (0, 0, 255))
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255))
        cv2.imshow("image", img)
    if event == cv2.EVENT_RBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        aoi_loc.append((x, y, 1))
        cv2.circle(img, (x, y), 2, (255, 0, 0))
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255))
        cv2.imshow("image", img)

img = cv2.imread(image_data)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Only left gaze data is used
x = np.array(data['x_cod_left_gaze'])     # x-coordinate
y = np.array(data['y_cod_left_gaze'])     # y-coordinate
time = np.array(data['time_stamp'])       # time stamp

timedur = st.sidebar.slider('Select the time interval', 1.0, 3.0)       # Slider used to choose the time interval.
integration_index = st.sidebar.number_input('Integration index', 3.0)
circle_size = 450-st.sidebar.slider('Choose the circle size', 150, 300)


# Fixation and saccade detection
Sfix, Efix = fixation_detection(x, y, time, missing=0.0, maxdist=15, mindur=50)
fixations = Efix
fixations_inte = fixations_integration(time=time, fixations=fixations, dur=integration_index)

#slice = fixations_slice(fixations_inte,dur=timedur)    #Get the slice location to cut fixations into pieces

fixation = draw_fixations(fixations, dispsize = [1920, 1080], imagefile=image_data,
                        alpha=0.4, size=circle_size,savefilename=bounding_addr)
fixation_inte = draw_fixations(fixations_inte, dispsize = [1920, 1080], imagefile=image_data,
                        alpha=0.4)
fixation = image_convert(fixation)
fixation_inte = image_convert(fixation_inte)


timeline = st.sidebar.slider('Timeline of fixations', 0, len(fixations_inte)-1)
click_state = st.sidebar.button('Click to get AOI position')


if click_state:
    cv2.namedWindow("image",cv2.WINDOW_FREERATIO)
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
    aoi_loc = []
    while (1):
        cv2.imshow("image", img)
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
    cv2.destroyAllWindows()
    click_state = False

fix_number = fix_count(fixations=fixations,AOI=aoi_loc)
bounding_map = draw_boundingbox(aoi=aoi_loc, imagefile=bounding_addr, fix_number=fix_number)

bs = list()
for j in range(len(fixations_inte)):
    fixation_slice = draw_fixations([fixations_inte[j]], dispsize = [1920, 1080],#savefilename=fixation_addr+'/%d.jpg' %j,
                        imagefile=image_data, alpha=0.4)
    #draw_arrows(fixations=fixations_inte, index=j, imagefile=fixation_addr+'/%d.jpg' %j)
    fixation_slice = image_convert(fixation_slice)
    bs.append(fixation_slice[np.newaxis, :])
fig_list = np.concatenate(bs, axis=0)   #fig_list contaning a list of figures, which are fixations map in different time

#generate fixation video
#video_generator(outname='fixation_output', slice=fixations_inte, input_addr=fixation_addr, fps=1/integration_index, size=(1920,1080))

step_list = step_count(fixations)
fix_dur, turnback_count, turnback_index = turnback_count(fixations)

st.header("Fixations")
st.image(fixation)

col1, col2 = st.columns(2)
with col1:
    st.header("Number of fixations in different AOI")
    st.bar_chart(fix_number)
with col2:
    st.header("AOI bounding map")
    st.image(bounding_map)

col3, col4 = st.columns(2)
with col3:
    st.header("Fixation in selected time")
    st.image(fig_list[timeline])
with col4:
    st.header("Transition Video")
    st.video(data=video_addr+'fixation_output.mp4')

st.header("Fixations transition map")
st.image(fixation_inte)

st.header("Step length line chart")
st.line_chart(step_list)

st.header("Duration of fixations")
st.line_chart(fix_dur)

st.header("Angles between two arrows")
st.line_chart(turnback_index)
