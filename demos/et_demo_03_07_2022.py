from psychopy import visual, event, core, gui, monitors
from psychopy.tools.monitorunittools import cm2deg

import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import pandas as pd
import random
import sys


import matplotlib
import os

from util import fixation_detection   # util file to find fixation and saccade detection.
from util import saccade_detection
from util import parse_fixations_window


from plots import draw_display
from plots import draw_raw
from plots import draw_fixations
from plots import draw_scanpath
from plots import draw_heatmap

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


from titta import Titta, helpers_tobii as helpers    
import tobii_research as tr



# ----- Read the data fron the eye tracker container -------------

def read_et_data():
    ''' Read eye tracking data from the buffer
    
    Returns:
        df - pandas dataframe
    '''
    df = pd.DataFrame(tracker.gaze_data_container, columns=tracker.header)
    df.to_csv('test1.csv')
    df.reset_index()
    
    return df

def stim_slideshow(fname, stim_type = 'text', 
                   duration=1, show_results=False):
    ''' Displays image stimuli in the folder 'fname'
    if fname is not a folder, a single image is displayed
    '''
    
    if 'image' in stim_type:
        ins = 'Explore the image \n(Press space to start).'
    else:
        ins = 'Read the text carefully. Press space when done reading \n(Press space to start).'
        
    instruction_text.setText(ins) # 
    instruction_text.draw()
    win.flip()
    #------------ Change the waiting time if needed -------------
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
        temp_im.size *= [2, 2]     # For full screen      
        im.append(temp_im)
        
        
    if eye_tracking:
        tracker.start_recording(gaze_data=True)
        tracker.gaze_data_container = [] # Remove calibration/validation data

        
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

        xy_l_pix = helpers.tobii2pix(xy_l, mon) # Conversion into pixel coordinates
        x = xy_l_pix[:, 0]
        y = xy_l_pix[:, 1]
        t = time

        # Fixation Detection
        Sfix, Efix = fixation_detection(x, y, t, missing=0.0, maxdist=50, mindur=50)
        fixations  = Efix 
        print('Length of the fixation:',len(Sfix))
           
        
        #fig = draw_raw(x, y, dispsize = [1920, 1080], imagefile='im', savefilename='raw.png')
  
        #df = read_et_data()
    
        im[i].draw()
        #fix = parse_fixations_window(fixations)
        #siz = 1 * (fix['dur']/300)
        #col = fix['dur']
        
       # plt.scatter(fix['x'],fix['y'], s=siz, c=col, marker='o', cmap='jet', alpha=0.5, edgecolors='none')

       # plt.gca().invert_yaxis()
       
        
        event.waitKeys()        
            




#%% Start of lab

# Select task
myDlg = gui.Dlg(title="Eye Tracking Data Collection")


myDlg.addField('Task', choices=["1. Images",
                                "2. Text"])


ok_data = myDlg.show()  # show dialog and wait for OK or Cancel
if myDlg.OK:  # or if ok_data is not None
    print(ok_data)
else:
    print('user cancelled')
    
 


# Sp parameters
amp = 0.8
nCycles = 2
cps = 0.2
pos = (-0.8, -amp*np.sin(nCycles / 2.0* 2*np.pi*-0.8))

stim_duration = 5

dot_positions_x = [-0.8, -0.4, 0, 0.4, 0.8]
dot_positions_y = [0,0,0,0,0]
random.shuffle(dot_positions_x)
dot_positions = list(zip(dot_positions_x, dot_positions_y))



#%% Monitor/geometry 
MY_MONITOR                  = 'dataMonitor' # needs to exists in PsychoPy monitor center
FULLSCREEN                  = True
SCREEN_RES                  = [1920, 1080]
SCREEN_WIDTH                = 52.7 # cm
SCREEN_HEIGHT               = 30.0 # cm
VIEWING_DIST                = 60 #  # distance from eye to center of screen (cm)

mon = monitors.Monitor(MY_MONITOR)  # Defined in defaults file
mon.setWidth(SCREEN_WIDTH)          # Width of screen (cm)
mon.setDistance(VIEWING_DIST)       # Distance eye / monitor (cm)
mon.setSizePix(SCREEN_RES)


# Change any of the default dettings?
settings = Titta.get_defaults(et_name)
settings.FILENAME = 'testfile.tsv'

# Switch to 1200 Hz if fixational eye movements
if 'fixation' in myDlg.data[0]:
    settings.SAMPLING_RATE = 1200
    
settings.mon = mon
settings.SCREEN_HEIGHT = SCREEN_HEIGHT





# Connect to eye tracker
tracker = Titta.Connect(settings)
if dummy_mode:
    tracker.set_dummy_mode()
tracker.init()

win = visual.Window(monitor = mon, fullscr = FULLSCREEN,
                    screen=1, size=SCREEN_RES, units = 'deg')
                    
win.size = SCREEN_RES      
print(win.size, SCREEN_RES)
tracker.calibrate(win)

  

dot = visual.Circle(win, radius = 0.01, fillColor='red', lineColor='white',
                    units='norm')
et_sample = visual.Circle(win, radius = 0.005, fillColor='red', lineColor='white',
                    units='norm')
text = visual.TextStim(win)
et_line = visual.Line(win, units='norm')
okn_stim = visual.GratingStim(win, color='black', tex='sqr',
                         sf = (0.5,0), mask=None,size=60)
instruction_text = visual.TextStim(win,color='black',text='',wrapWidth = 20,height = 1)

#if 'Static dots' in myDlg.data[0]:
#    present_dots(dot_positions, duration=1, show_results=True)    
#elif 'fixation' in myDlg.data[0]:
#    present_dots([[0, 0]], duration=10, show_results=True)        
if 'Images' in myDlg.data[0]:
    # display_fixation_cross()
    files = glob.glob(os.getcwd() + os.sep + 'images' + os.sep + '*.bmp')
    random.shuffle(files)
    stim_slideshow(files[0], duration=stim_duration, show_results=True, 
                   stim_type = 'image')
    #stim_slideshow('images', duration=stim_duration, show_results=True, 
    #               stim_type = 'image') 
elif 'Text' in myDlg.data[0]:
    # display_fixation_cross()
    files = glob.glob(os.getcwd() + os.sep + 'texts' + os.sep + '*.bmp')
    random.shuffle(files)
    stim_slideshow(files[0], duration=None, show_results=True,
                   stim_type = 'text')
 

plt.close('all')
    
# Stop eye tracker and clean up 
if eye_tracking:
    #tracker.stop_sample_buffer()
    tracker.stop_recording(gaze_data=True)
    tracker.de_init()    
    
#win.close()
#core.quit()

    
