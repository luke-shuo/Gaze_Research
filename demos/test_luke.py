1  # -*- coding: utf-8 -*-
"""
Before running
1) Change the 'et_name' below to the eye tracker model you are using.
2) Make sure the monitor settings (in 'mon') are aligned with your specific setup.
Then select from the drop down menu which task you want to run. Data will be presented 
immediately after the recording is done.
"""

from psychopy import visual, event, core, gui, monitors
from psychopy.tools.monitorunittools import cm2deg
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import pandas as pd
import random
import sys
import csv
import matplotlib
from util import fixation_detection   # util file to find fixation and saccade detection.
from util import saccade_detection
from util import parse_fixations_window
from plots import draw_display
from plots import draw_raw
from plots import draw_fixations
from plots import draw_scanpath
from plots import draw_heatmap
import cv2
from titta import Titta, helpers_tobii as helpers
import tobii_research as tr

# UDP libs
import socket
import pickle
class udp_client():
    def __init__(self, ip, port):
        self.serverAddressPort   = (ip,port)
        self.bufferSize          = 1024
        self.UDPClientSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    def message(self,msg):
        #msgstr = str(msg)
        #bytes = str.encode(msgstr)
        bytes =pickle.dumps(msg)
        self.UDPClientSocket.sendto(bytes, self.serverAddressPort)

def plot_et_data(df, plot_type='xy'):
    ''' Plots eye-tracker data as xy or xy over time.

    Args:
        df - pandas dataframe with eye-tracking data
    '''

    xy_l = np.array(df[['left_gaze_point_on_display_area_x',
                        'left_gaze_point_on_display_area_y']])
    xy_l = helpers.tobii2deg(xy_l, settings.mon)
    # xy_l = helpers.tobii2pix(xy_l, settings.mon)

    xy_r = np.array(df[['right_gaze_point_on_display_area_x',
                        'right_gaze_point_on_display_area_y']])
    xy_r = helpers.tobii2deg(xy_r, settings.mon)
    # xy_r = helpers.tobii2pix(xy_r, settings.mon)

    # Lowpass filter if fixational eye movements
    if plot_type == 'xy':
        plt.plot(xy_l[:, 0],
                 xy_l[:, 1], '.', ms=2, c='r', label='Left eye')
        plt.plot(xy_r[:, 0],
                 xy_r[:, 1], '.', ms=2, c='b', label='Right eye')
        plt.xlabel('Horizontal gaze coordinate (deg)')
        plt.ylabel('Vertical gaze coordinate (deg)')
        plt.legend()

        axis_deg = helpers.tobii2deg(np.array([[0.99, 0.99]]), settings.mon).flatten()
        plt.axis([-axis_deg[0], axis_deg[0], -axis_deg[1],
                  axis_deg[1]])
        plt.gca().invert_yaxis()
    else:
        tt = np.array(df['system_time_stamp'])
        #        print(tt)
        tt = (tt - tt[0]) / 1000.0 / 1000.0
        plt.plot(tt,
                 xy_r[:, 0], '-', label='Horizontal gaze coordinate (right eye)')
        plt.plot(tt,
                 xy_r[:, 1], '-', label='Vertical gaze coordinate (right eye)')
        plt.xlabel('Time (s)')
        plt.legend()
        plt.ylabel('Horizontal/vertical gaze coordinate (deg)')
        # plt.show()

def key_generator():
    # Modified code for generating the keys Read EYE TRACKER DATA
    et_data = pd.read_csv('test1.csv')
    # et_data

    # Data extraction interms of coordinate
    xy_l = np.array(et_data[['left_gaze_point_on_display_area_x',
                             'left_gaze_point_on_display_area_y']])
    xy_r = np.array(et_data[['right_gaze_point_on_display_area_x',
                             'right_gaze_point_on_display_area_y']])

    # time = np.array(et_data['device_time_stamp'])

    xy_l_pix = helpers.tobii2pix(xy_l, mon)  # Conversion into pixel coordinates
    xy_l = pd.DataFrame(xy_l_pix)
    xy_l.to_csv('left_pix.csv')
    xy_r_pix = helpers.tobii2pix(xy_r, mon)  # Conversion into pixel coordinates
    # print(xy_l_pix)
    # print(xy_r_pix)

    # Sample collection

    x_cod = xy_l_pix[:, 0]  # first 5 seconds
    # x_cod1 = xy_l_pix[:, 0]     # Next 5 seconds

    y_cod = xy_l_pix[:, 1]
    # y_cod1 = xy_l_pix[301:601, 1]

    aoi_1_topleft_x = 480  # Coordinates of area of interest 1
    aoi_1_topleft_y = 430
    aoi_1_bottomright_x = 730
    aoi_1_bottomright_y = 820

    aoi_2_topleft_x = 1080  # Coordinates for area of Interest 2
    aoi_2_topleft_y = 470
    aoi_2_bottomright_x = 1250
    aoi_2_bottomright_y = 760

    no_sample_aoi1 = []
    no_sample_aoi2 = []

    for i in range(len(x_cod)):
        if ((x_cod[i] >= aoi_1_topleft_x and x_cod[i] <= aoi_1_bottomright_x) and
                (y_cod[i] >= aoi_1_topleft_y and y_cod[i] <= aoi_1_bottomright_y)):
            no_sample_aoi1.append(x_cod[i])
        elif ((x_cod[i] >= aoi_2_topleft_x and x_cod[i] <= aoi_2_bottomright_x) and
              (y_cod[i] >= aoi_2_topleft_y and y_cod[i] <= aoi_2_bottomright_y)):
            no_sample_aoi2.append(x_cod[i])

    # -------------------------------- Percentage Samples ------------------------------------

    per_aoi1 = len(no_sample_aoi1) / len(x_cod)
    per_aoi2 = len(no_sample_aoi2) / len(x_cod)

    # ------------------------ Generating key for operation ---------------------------------

    key1 = 0
    key2 = 0
    if (per_aoi1 >= 0.20):
        key1 = 1

    if (per_aoi2 >= 0.20):
        key2 = 2

    # --------------------------------- Generating Message UDP ------------------------
    udp = udp_client('192.168.8.113', 20002)
    key = [key1, key2]
    print(key)
    udp.message(key)

def read_et_data():
    ''' Read eye tracking data from the buffer

    Returns:
        df - pandas dataframe
    '''
    df = pd.DataFrame(tracker.gaze_data_container, columns=tracker.header)
    df.to_csv('test1.csv')
    df.reset_index()
    return df

def stim_slideshow(fname, stim_type='text', duration=1, show_results=False):
    ''' Displays image stimuli in the folder 'fname'
    if fname is not a folder, a single image is displayed
    '''

    if 'image' in stim_type:
        ins = 'Stare one of the object apears on the screen \n(Press space to start).'
    else:
        ins = 'Read the text carefully. Press space when done reading \n(Press space to start).'

    instruction_text.setText(ins)  #
    instruction_text.draw()
    win.flip()
    k = event.waitKeys()
    if k[0] == 'q':
        win.close()
        core.quit()

    if os.path.isdir(fname):
        # List pictures
        stimnames = glob.glob(fname)

    else:
        stimnames = [fname]

    # preload pictures
    im = []
    for stim in stimnames:
        temp_im = visual.ImageStim(win, image=stim,
                                   units='norm', size=(1, 1))
        temp_im.size *= [2, 2]  # For full screen
        im.append(temp_im)

    if eye_tracking:
        tracker.start_recording(gaze_data=True)
        tracker.gaze_data_container = []  # Remove calibration/validation data

    # Show pictures
    for i in range(len(im)):
        im[i].draw()
        win.flip()
        if not duration:
            event.waitKeys()
        else:
            core.wait(duration)

        key = event.getKeys()
        if 'escape' in key:
            win.close()
            core.quit()

    win.flip()

    if eye_tracking:
        tracker.stop_recording(gaze_data=True)

    if eye_tracking and show_results:
        et_data = read_et_data()

        xy_l = np.array(et_data[['left_gaze_point_on_display_area_x',
                                 'left_gaze_point_on_display_area_y']])
        xy_r = np.array(et_data[['right_gaze_point_on_display_area_x',
                                 'right_gaze_point_on_display_area_y']])

        time = np.array(et_data['device_time_stamp'])

        xy_l_pix = helpers.tobii2pix(xy_l, mon)  # Conversion into pixel coordinates
        x = xy_l_pix[:, 0]
        y = xy_l_pix[:, 1]
        t = time

        # Fixation Detection
        Sfix, Efix = fixation_detection(x, y, t, missing=0.0, maxdist=50, mindur=50)
        fixations = Efix
        # print(len(Sfix))

        # Saccade Detection

        Ssac, Esac = saccade_detection(x, y, t, missing=0.0, minlen=50, maxvel=300, maxacc=400)
        saccades = Esac

        fig = draw_raw(x, y, dispsize=[1920, 1080], imagefile='1.png', savefilename='raw.png')
        heatmap = draw_heatmap(fixations, dispsize=[1920, 1080], imagefile='1.png',
                               durationweight=True, alpha=0.4, savefilename='heat')

        # create figure

        # img1=cv2.imread('heat.png')
        # plt.title('Raw_Data')
        # plt.imshow(img1)
        # plt.show()

        # xy_l= pd.DataFrame(xy_l_pix)
        # xy_l.to_csv('left_pix.csv')
        # xy_r_pix = helpers.tobii2pix(xy_r, mon)
        key_generator()  # For Robotic Control

        event.waitKeys()


# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)
sys.path.insert(0, os.path.dirname(_thisDir))
plt.close('all')
print(os.getcwd())

# Set parameters
eye_tracking = True
et_name = 'Tobii Pro Nano'
dummy_mode = False

# start lab
# Select task
myDlg = gui.Dlg(title="Gaze Aware Robotic Control Tests")
# myDlg.addField('Task', choices=["Pursuit","Pursuit (blank screen)", "Static dots",
#                                 "images", "text", "okn"])

myDlg.addField('Task', choices=["Images"])
# "2. Text"])

ok_data = myDlg.show()  # show dialog and wait for OK or Cancel
if myDlg.OK:  # or if ok_data is not None
    print(ok_data)
else:
    print('user cancelled')

# Sp parameters
amp = 0.8
nCycles = 2
cps = 0.2
pos = (-0.8, -amp * np.sin(nCycles / 2.0 * 2 * np.pi * -0.8))

stim_duration = 10

dot_positions_x = [-0.8, -0.4, 0, 0.4, 0.8]
dot_positions_y = [0, 0, 0, 0, 0]
# random.shuffle(dot_positions_x)
dot_positions = list(zip(dot_positions_x, dot_positions_y))

# print(dot_positions)
# random.shuffle(dot_positions)
# print(dot_positions)

# %% Monitor/geometry
MY_MONITOR = 'testMonitor'  # needs to exists in PsychoPy monitor center
FULLSCREEN = True
# SCREEN_RES                  = [3840, 2160]
SCREEN_RES = [1920, 1080]
SCREEN_WIDTH = 30.937  # cm
SCREEN_HEIGHT = 17.377  # cm
VIEWING_DIST = 40  # # distance from eye to center of screen (cm)

mon = monitors.Monitor(MY_MONITOR)  # Defined in defaults file
mon.setWidth(SCREEN_WIDTH)  # Width of screen (cm)
mon.setDistance(VIEWING_DIST)  # Distance eye / monitor (cm)
mon.setSizePix(SCREEN_RES)

# Change any of the default dettings?
settings = Titta.get_defaults(et_name)
settings.FILENAME = 'testfile.tsv'
# settings.RECORD_EYE_IMAGES_DURING_CALIBRATION = True

settings.mon = mon
settings.SCREEN_HEIGHT = SCREEN_HEIGHT

# Connect to eye tracker
tracker = Titta.Connect(settings)
if dummy_mode:
    tracker.set_dummy_mode()
tracker.init()

win = visual.Window(monitor = mon, fullscr = FULLSCREEN,
                    #screen=1, size=SCREEN_RES, units = 'deg')

#win = visual.Window(monitor = mon, fullscr = FULLSCREEN,
                     size=SCREEN_RES, units = 'deg')

##win.size = SCREEN_RES
#print(win.size, SCREEN_RES)

tracker.calibrate(win)

dot = visual.Circle(win, radius=0.05, fillColor='red', lineColor='white',
                    units='norm')
et_sample = visual.Circle(win, radius=0.005, fillColor='red', lineColor='white',
                          units='norm')
text = visual.TextStim(win)
et_line = visual.Line(win, units='norm')
okn_stim = visual.GratingStim(win, color='black', tex='sqr',
                              sf=(0.5, 0), mask=None, size=60)
instruction_text = visual.TextStim(win, color='black', text='', wrapWidth=20, height=1)

## Show instructions
text.setText('Press space to start')
# text.draw()
# win.flip()
# event.waitKeys()


if '4. Pursuit' in myDlg.data[0]:
    #    display_cue(pos)
    sinusoid_pursuit(nCycles=nCycles, cps=cps, amp=amp, show_results=True, blank_screen=False)
elif '5. Pursuit' in myDlg.data[0]:
    #    display_cue(pos)
    sinusoid_pursuit(nCycles=nCycles, cps=cps, amp=amp, show_results=True, blank_screen=True)
elif 'Static dots' in myDlg.data[0]:
    present_dots(dot_positions, duration=1, show_results=True)
elif 'fixation' in myDlg.data[0]:
    present_dots([[0, 0]], duration=10, show_results=True)
elif 'Images' in myDlg.data[0]:
    # display_fixation_cross()
    files = glob.glob(os.getcwd() + os.sep + 'images' + os.sep + '*.png')
    random.shuffle(files)
    stim_slideshow(files[0], duration=stim_duration, show_results=True,
                   stim_type='image')
elif 'Text' in myDlg.data[0]:
    # display_fixation_cross()
    files = glob.glob(os.getcwd() + os.sep + 'texts' + os.sep + '*.bmp')
    random.shuffle(files)
    stim_slideshow(files[0], duration=None, show_results=True,
                   stim_type='text')
else:
    temporal_frequency = 5.0
    okn_dur = 5
    screen_Fs = 60.0

    okn(temporal_frequency, okn_dur, screen_Fs, direction='R', show_instruction=True)

plt.close('all')

# Stop eye tracker and clean up
if eye_tracking:
    # tracker.stop_sample_buffer()
    tracker.stop_recording(gaze_data=True)
    tracker.de_init()

# win.close()
# core.quit()