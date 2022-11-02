import time
import tobii_research as tr
from tobii_research_addons import ScreenBasedCalibrationValidation, Point2

eyetracker_address = 'Replace the address of the desired tracker'
eyetracker = tr.EyeTracker(eyetracker_address)