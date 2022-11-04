import streamlit as st
import pandas as pd
import numpy as np
import cv2
import os
import graphviz
import globalVal
from plots import draw_fixations
from plots import image_convert
from plots import draw_aoi_map
from plots import draw_boundingbox
from util import fixation_detection   # util file to find fixation and saccade detection.
from util import find_dominate_aoi
from util import find_steppulse_index
from util import step_count
from util import fix_count
from util import jump_aoiLocation
from util import remove_overlap
from util import checkOverlap

imageFileName = []
for filename in os.listdir(globalVal.images_path):
    imageFileName.append(filename[-8:-5])
image_name = st.sidebar.selectbox('Select the image',imageFileName)
image_csv = pd.read_csv(globalVal.dataset_path+'image.csv')
image_list = np.array(image_csv['0']).tolist()
dataset_index = image_list.index(image_name)

dataset = globalVal.dataset_path + 'dataset%d.csv' % dataset_index
image = globalVal.images_path + image_name + '.jpeg'

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

period = st.sidebar.selectbox('Choose the period of each piece',(10,5))
fig_index = st.sidebar.slider('Timeline of fixations', 1, int(60/period))

img = cv2.imread(image)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

@st.cache
def load_data():
    data = pd.read_csv(dataset)
    data.fillna(0, inplace=True)    # Using '0' to replace 'N\A'
    return data
data = load_data()

x = np.array(data['x_cod_left_gaze'])     # x-coordinate
y = np.array(data['y_cod_left_gaze'])     # y-coordinate
time = np.array(data['time_stamp'])       # time stamp

dominate_aoi = []
bs = list()
for i in range(int(60/period)):
    Sfix, Efix = fixation_detection(x[i*60*period:(i+1)*60*period], y[i*60*period:(i+1)*60*period], time[i*60*period:(i+1)*60*period], missing=0.0, maxdist=15, mindur=50)
    fixations = Efix
    fixation = draw_fixations(fixations, dispsize=[1920, 1080], imagefile=image,
                              alpha=0.4)
    fixation = image_convert(fixation)
    bs.append(fixation[np.newaxis, :])
    dominate_aoi.append(find_dominate_aoi(fixations=fixations, aoi_num=(60 / period)))


Sfix, Efix = fixation_detection(x, y, time, missing=0.0, maxdist=15, mindur=50)
fixations = Efix

step_list = step_count(fixations)
step_index = find_steppulse_index(step_list)

# find aoi based on fixation jump
aoi_loc = jump_aoiLocation(step_index,fixations)
fix_number = fix_count(fixations=fixations,AOI=aoi_loc)
bounding_map = draw_boundingbox(aoi=aoi_loc, imagefile=image, fix_number=fix_number)

while True:
    # check overlap
    state = checkOverlap(aoi_loc)
    # remove overlap
    aoi_loc = remove_overlap(aoi_loc,state)
    state = checkOverlap(aoi_loc)
    if sum(state) == 0:
        break

map_overlap = draw_boundingbox(aoi=aoi_loc, imagefile=image, fix_number=fix_number)


fig_index_impulse = st.sidebar.slider('Timeline of fixations based on impulse', 1, len(step_index)-1)

dominate_aoi_impulse = []
bs_impulse = list()
for i in range(len(step_index)-1):
    fixation = draw_fixations(fixations[step_index[i]:step_index[i+1]], dispsize=[1920, 1080], imagefile=image,
                              alpha=0.4)
    fixation = image_convert(fixation)
    bs_impulse.append(fixation[np.newaxis, :])
    dominate_aoi_impulse.append(find_dominate_aoi(fixations=fixations[step_index[i]:step_index[i+1]], aoi_num=(60 / period)))

graph = graphviz.Digraph()
for i in range(len(dominate_aoi)-1):
    graph.edge('AOI %d' % dominate_aoi[i], 'AOI %d' % dominate_aoi[i+1])

graph_impulse = graphviz.Digraph()
for i in range(len(dominate_aoi_impulse) - 1):
    graph_impulse.edge('AOI %d' % dominate_aoi_impulse[i], 'AOI %d' % dominate_aoi_impulse[i + 1])

dominate_aoi = np.array(dominate_aoi).reshape(1,int(60/period))
df = pd.DataFrame(dominate_aoi, columns=('Time %d' % i for i in range(1,int(60/period)+1)))

dominate_aoi_impulse = np.array(dominate_aoi_impulse).reshape(1, len(dominate_aoi_impulse))
df_impulse = pd.DataFrame(dominate_aoi_impulse, columns=('Impulse %d' % i for i in range(1, len(step_index))))

fig_list = np.concatenate(bs, axis=0)
fig_list_impulse = np.concatenate(bs_impulse, axis=0)
draw_aoi_map(img,period)


st.header('AOI map')
st.image(img)

col1, col2 = st.columns(2)
with col1:
    st.header('Fixation map in selected time')
    st.image(fig_list[fig_index - 1])
with col2:
    st.header('Fixation map in selected impulse')
    st.image(fig_list_impulse[fig_index_impulse -1])

col5, col6= st.columns(2)
with col5:
    st.header("AOI map based on fixation jump")
    st.image(bounding_map)
with col6:
    st.header("AOI map without overlap")
    st.image(map_overlap)

st.header('Dominate AOI in selected time')
st.table(df)
st.header('Dominate AOI in selected impulse')
st.table(df_impulse)


col3, col4 = st.columns(2)
with col3:
    st.header('State Machine based on time')
    st.graphviz_chart(graph)
with col4:
    st.header('State Machine based on impulse')
    st.graphviz_chart(graph_impulse)
