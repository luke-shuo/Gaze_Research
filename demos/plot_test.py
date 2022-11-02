from psychopy import visual, event, core, gui, monitors
from psychopy.tools.monitorunittools import cm2deg

import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import pandas as pd
import random
import sys


# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)
sys.path.insert(0, os.path.dirname(_thisDir))



plt.close('all')
print(os.getcwd())

eye_tracking = True
et_name = 'Tobii Pro Nano'
dummy_mode = False


from titta import Titta, helpers_tobii as helpers    
import tobii_research as tr

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


settings.mon = mon
settings.SCREEN_HEIGHT = SCREEN_HEIGHT

win = visual.Window(monitor = mon, fullscr = FULLSCREEN,
                    screen=1, size=SCREEN_RES, units = 'deg')
                    
win.size = SCREEN_RES      
print(win.size, SCREEN_RES)
#tracker.calibrate(win)



dot = visual.Circle(win, radius = 0.01, fillColor='red', lineColor='white',
                    units='norm')
et_sample = visual.Circle(win, radius = 0.005, fillColor='red', lineColor='white',
                    units='norm')
text = visual.TextStim(win)

et_line = visual.Line(win, units='norm')

instruction_text = visual.TextStim(win,color='black',text='',wrapWidth = 20,height = 1)


files = glob.glob(os.getcwd() + os.sep + 'images' + os.sep + '*.bmp')
#random.shuffle(files)

fname = files[0]
duration = False

print('fname', fname)


if os.path.isdir(fname):
    stimnames = glob.glob(fname)
else:
    stimnames = [fname]
    
im = []

for stim in stimnames:
    temp_im = visual.ImageStim(win, image=stim, units='norm', size=(1, 1))
    temp_im.size *= [2, 2]     # For full screen      
    im.append(temp_im)
for i in range(len(im)):
    plt.figure()
    plt.subplot(2,1,1)
    im[i].draw()
    
    event.waitKeys()
    win.flip()
    
    
    
    
    #m[i].draw()
   #win.flip()
    #f not duration:
   #    event.waitKeys()
   #else:
  #     core.wait(duration)
   #key = event.getKeys()
   #if 'escape' in key:
   #    win.close()
   #    core.quit()
        
        
          
#win.flip()




    
