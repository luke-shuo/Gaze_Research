import os
# globalVal.py
# Used to store global value

# Data address
if os.name == 'nt':
    dataCollector_path = 'C:\\Users\\2602651K\\Documents\\GitHub\\Gaze_Research\\dashboard\\dataCollector\\'
    images_path = 'C:\\Users\\2602651K\\Documents\\GitHub\\Gaze_Research\\dashboard\\dataCollector\\images\\'
    videos_path = 'C:\\Users\\2602651K\\Documents\\GitHub\\Gaze_Research\\dashboard\\dataCollector\\videos\\'
    dataset_path = 'C:\\Users\\2602651K\\Documents\\GitHub\\Gaze_Research\\dashboard\\dataset\\'
    fixations_path = 'C:\\Users\\2602651K\\Documents\\GitHub\\Gaze_Research\\dashboard\\fixations\\'
    video_output_path = 'C:\\Users\\2602651K\\Documents\\GitHub\\Gaze_Research\\dashboard\\video_output\\'
    bounding_image_path = 'C:\\Users\\2602651K\\Documents\\GitHub\\Gaze_Research\\dashboard\\bounding_image\\'
else:
<<<<<<< HEAD
    dataCollector_path = '/Users/lukeshuo/Documents/GitHub/Gaze_Research/dashboard/dataCollector/'
    images_path = '/Users/lukeshuo/Documents/GitHub/Gaze_Research/dashboard/dataCollector/images/'
    videos_path = '/Users/lukeshuo/Documents/GitHub/Gaze_Research/dashboard/dataCollector/videos/'
    dataset_path = '/Users/lukeshuo/Documents/GitHub/Gaze_Research/dashboard/dataset/'
    fixations_path = '/Users/lukeshuo/Documents/GitHub/Gaze_Research/dashboard/fixations/'
    video_output_path = '/Users/lukeshuo/Documents/GitHub/Gaze_Research/dashboard/video_output/'
    bounding_image_path = '/Users/lukeshuo/Documents/GitHub/Gaze_Research/dashboard/bounding_image/'

