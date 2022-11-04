import math
import cv2
import numpy as np
import pandas as pd
import globalVal

video_addr = globalVal.video_output_path

# --------------Functions to detect the fixation and saccade detection ------------------
def blink_detection(x, y, time, missing=0.0, minlen=10):
    """Detects blinks, defined as a period of missing data that lasts for at
    least a minimal amount of samples

    arguments
    x		-	numpy array of x positions
    y		-	numpy array of y positions
    time		-	numpy array of EyeTribe timestamps
    keyword arguments
    missing	-	value to be used for missing data (default = 0.0)
    minlen	-	integer indicating the minimal amount of consecutive
                missing samples

    returns
    Sblk, Eblk
                Sblk	-	list of lists, each containing [starttime]
                Eblk	-	list of lists, each containing [starttime, endtime, duration]
    """

    # empty list to contain data
    Sblk = []
    Eblk = []

    # check where the missing samples are
    mx = np.array(x == missing, dtype=int)
    my = np.array(y == missing, dtype=int)
    miss = np.array((mx + my) == 2, dtype=int)

    # check where the starts and ends are (+1 to counteract shift to left)
    diff = np.diff(miss)
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    # compile blink starts and ends
    for i in range(len(starts)):
        # get starting index
        s = starts[i]
        # get ending index
        if i < len(ends):
            e = ends[i]
        elif len(ends) > 0:
            e = ends[-1]
        else:
            e = -1
        # append only if the duration in samples is equal to or greater than
        # the minimal duration
        if e - s >= minlen:
            # add starting time
            Sblk.append([time[s]])
            # add ending time
            Eblk.append([time[s], time[e], time[e] - time[s]])

    return Sblk, Eblk


def remove_missing(x, y, time, missing):
    mx = np.array(x == missing, dtype=int)
    my = np.array(y == missing, dtype=int)
    x = x[(mx + my) != 2]
    y = y[(mx + my) != 2]
    time = time[(mx + my) != 2]
    return x, y, time


def fixation_detection(x, y, time, missing=0.0, maxdist=25, mindur=50):
    """Detects fixations, defined as consecutive samples with an inter-sample
    distance of less than a set amount of pixels (disregarding missing data)

    arguments
    x		-	numpy array of x positions
    y		-	numpy array of y positions
    time		-	numpy array of EyeTribe timestamps
    keyword arguments
    missing	-	value to be used for missing data (default = 0.0)
    maxdist	-	maximal inter sample distance in pixels (default = 25)
    mindur	-	minimal duration of a fixation in milliseconds; detected
                fixation cadidates will be disregarded if they are below
                this duration (default = 100)

    returns
    Sfix, Efix
                Sfix	-	list of lists, each containing [starttime]
                Efix	-	list of lists, each containing [starttime, endtime, duration, endx, endy]
    """

    x, y, time = remove_missing(x, y, time, missing)

    # empty list to contain data
    Sfix = []
    Efix = []

    # loop through all coordinates
    si = 0
    fixstart = False
    for i in range(1, len(x)):
        # calculate Euclidean distance from the current fixation coordinate
        # to the next coordinate
        squared_distance = ((x[si] - x[i]) ** 2 + (y[si] - y[i]) ** 2)
        dist = 0.0
        if squared_distance > 0:
            dist = squared_distance ** 0.5
        # check if the next coordinate is below maximal distance
        if dist <= maxdist and not fixstart:
            # start a new fixation
            si = 0 + i
            fixstart = True
            Sfix.append([time[i]])
        elif dist > maxdist and fixstart:
            # end the current fixation
            fixstart = False
            # only store the fixation if the duration is ok
            if time[i - 1] - Sfix[-1][0] >= (mindur*1000):  #used to be mindur
                Efix.append([Sfix[-1][0], time[i - 1], time[i - 1] - Sfix[-1][0], x[si], y[si]])
            # delete the last fixation start if it was too short
            else:
                Sfix.pop(-1)
            si = 0 + i
        elif not fixstart:
            si += 1
    # add last fixation end (we can lose it if dist > maxdist is false for the last point)
    if len(Sfix) > len(Efix):
        Efix.append([Sfix[-1][0], time[len(x) - 1], time[len(x) - 1] - Sfix[-1][0], x[si], y[si]])
    return Sfix, Efix


def saccade_detection(x, y, time, missing=0.0, minlen=5, maxvel=40, maxacc=340):
    """Detects saccades, defined as consecutive samples with an inter-sample
    velocity of over a velocity threshold or an acceleration threshold

    arguments
    x		-	numpy array of x positions
    y		-	numpy array of y positions
    time		-	numpy array of tracker timestamps in milliseconds
    keyword arguments
    missing	-	value to be used for missing data (default = 0.0)
    minlen	-	minimal length of saccades in milliseconds; all detected
                saccades with len(sac) < minlen will be ignored
                (default = 5)
    maxvel	-	velocity threshold in pixels/millisecond (default = 40)
    maxacc	-	acceleration threshold in pixels / millisecond**2
                (default = 340)

    returns
    Ssac, Esac
            Ssac	-	list of lists, each containing [starttime]
            Esac	-	list of lists, each containing [starttime, endtime, duration, startx, starty, endx, endy]
    """
    x, y, time = remove_missing(x, y, time, missing)

    # CONTAINERS
    Ssac = []
    Esac = []

    # INTER-SAMPLE MEASURES
    # the distance between samples is the square root of the sum
    # of the squared horizontal and vertical interdistances
    intdist = (np.diff(x) ** 2 + np.diff(y) ** 2) ** 0.5
    # get inter-sample times
    inttime = np.diff(time)
    # recalculate inter-sample times to seconds
    inttime = inttime / 1000.0

    # VELOCITY AND ACCELERATION
    # the average velocity between samples is the inter-sample distance
    # divided by the inter-sample time
    vel = intdist / inttime
    # the acceleration is the sample-to-sample difference in
    # eye movement velocity
    acc = np.diff(vel)

    # SACCADE START AND END
    t0i = 0
    stop = False
    while not stop:
        # saccade start (t1) is when the velocity or acceleration
        # surpass threshold, saccade end (t2) is when both return
        # under threshold

        # detect saccade starts
        sacstarts = np.where((vel[1 + t0i:] > maxvel).astype(int) + (acc[t0i:] > maxacc).astype(int) >= 1)[0]
        if len(sacstarts) > 0:
            # timestamp for starting position
            t1i = t0i + sacstarts[0] + 1
            if t1i >= len(time) - 1:
                t1i = len(time) - 2
            t1 = time[t1i]

            # add to saccade starts
            Ssac.append([t1])

            # detect saccade endings
            sacends = np.where((vel[1 + t1i:] < maxvel).astype(int) + (acc[t1i:] < maxacc).astype(int) == 2)[0]
            if len(sacends) > 0:
                # timestamp for ending position
                t2i = sacends[0] + 1 + t1i + 2
                if t2i >= len(time):
                    t2i = len(time) - 1
                t2 = time[t2i]
                dur = t2 - t1

                # ignore saccades that did not last long enough
                if dur >= minlen:
                    # add to saccade ends
                    Esac.append([t1, t2, dur, x[t1i], y[t1i], x[t2i], y[t2i]])
                else:
                    # remove last saccade start on too low duration
                    Ssac.pop(-1)

                # update t0i
                t0i = 0 + t2i
            else:
                stop = True
        else:
            stop = True

    return Ssac, Esac


def parse_fixations_window(fixations):
    """Returns all relevant data from a list of fixation ending events

    arguments

    fixations		-	a list of fixation ending events from a single trial,
                    as produced by edfreader.read_edf, e.g.
                    edfdata[trialnr]['events']['Efix']
    returns

    fix		-	a dict with three keys: 'x', 'y', and 'dur' (each contain
                a numpy array) for the x and y coordinates and duration of
                each fixation
    """

    # empty arrays to contain fixation coordinates
    fix = {'x': np.zeros(len(fixations)),
           'y': np.zeros(len(fixations)),
           'dur': np.zeros(len(fixations)),
           'st': np.zeros(len(fixations)),
           'et': np.zeros(len(fixations))}
    # get all fixation coordinates
    for fixnr in range(len(fixations)):
        stime, etime, dur, ex, ey = fixations[fixnr]
        fix['x'][fixnr] = ex
        fix['y'][fixnr] = ey
        fix['dur'][fixnr] = dur
        fix['st'][fixnr] = stime
        fix['et'][fixnr] = etime
    return fix


# ----------------------Helper function to convert the fixation data in the reasonable form
def parse_fixations(fixations):
    """Returns all relevant data from a list of fixation ending events

    arguments

    fixations		-	a list of fixation ending events from a single trial,
                    as produced by edfreader.read_edf, e.g.
                    edfdata[trialnr]['events']['Efix']
    returns

    fix		-	a dict with three keys: 'x', 'y', and 'dur' (each contain
                a numpy array) for the x and y coordinates and duration of
                each fixation
    """

    # empty arrays to contain fixation coordinates
    fix = {'x': np.zeros(len(fixations)),
           'y': np.zeros(len(fixations)),
           'dur': np.zeros(len(fixations))}
    # get all fixation coordinates
    for fixnr in range(len(fixations)):
        stime, etime, dur, ex, ey = fixations[fixnr]
        fix['x'][fixnr] = ex
        fix['y'][fixnr] = ey
        fix['dur'][fixnr] = dur

    return fix


def raw_data_slide(raw_data):
    raw_data_s1 = raw_data[raw_data['t'] <= 157 * fact]  # time < 157 seconds
    raw_data_s2 = raw_data[raw_data['t'].between(157 * fact, 340 * fact, inclusive=False)]  # time >157 < 340 seconds
    raw_data_s3 = raw_data[raw_data['t'] > 340 * fact]  # time > 340 seconds
    return raw_data_s1, raw_data_s2, raw_data_s3


def fix_count(fixations, AOI):
    """Counting how many fixations inward or outward the AOI

    arguments

    fixations	-	a list of fixation ending events from a single trial,
                    as produced by edfreader.read_edf, e.g.
                    edfdata[trialnr]['events']['Efix']
    AOI		-	Location of AOI containing top_left position and bottom_right position
                e.g. [(x1, y1, 0), (x2, y2, 1)];
                (x1, y1, 0) is the top_left position;
                (x2, y2, 1) is the bottom_right position
                if the input top_left and bottom_right position are like this, [(x1, y1, 0), (x2, y2, 1)], the fixations
                will be counted within this AOI. However, if the input positions are like this,  [(x1, y1, 1), (x2, y2, 0)],
                the fixations will be counted outward of this AOI.
    returns

    fix_number 	-	numbers of fixations within the AOI
    """

    fix = parse_fixations(fixations)
    x_loc = fix['x']
    y_loc = fix['y']
    dur = fix['dur']

    fix_num = []
    for j in range(0, len(AOI), 2):
        top_left = (AOI[j][0], AOI[j][1])
        bottom_right = (AOI[j + 1][0], AOI[j + 1][1])
        inward = AOI[j][2] - AOI[j + 1][2]

        fix_number = 0

        if inward == -1:
            for i in range(len(fixations)):
                # Fixation inward AOI
                if top_left[0] <= x_loc[i] <= bottom_right[0] and top_left[1] <= y_loc[i] <= bottom_right[1]:
                    fix_number = fix_number + 1
                else:
                    fix_number = fix_number
        else:
            for i in range(len(fixations)):
                # Fixation outward AOI
                if top_left[0] <= x_loc[i] <= bottom_right[0] and top_left[1] <= y_loc[i] <= bottom_right[1]:
                    fix_number = fix_number
                else:
                    fix_number = fix_number + 1
        fix_num.append(fix_number)
    fix_num = np.array(fix_num)  # count fixations in specific AOI
    return fix_num


def fixations_slice(fixations, dur=5):
    """Cut fixations into several slots with the same duration

    arguments

    fixations	-	a list of fixation ending events from a single trial,
                    as produced by edfreader.read_edf, e.g.
                    edfdata[trialnr]['events']['Efix']

    dur	-	control the slot duration
                default is 5 seconds
    returns

    slice 	-	a list of positions indicates where the generated slots are. e.g. [0, 100, 232, 600], which means there
                are 3 slots, fixations[0:100], fixations[100:232], fixations[232:600], are generated.
    """

    t = dur * 1000000
    count = 0
    slice = [0]
    for i in range(len(fixations)):
        count = count + fixations[i][2]
        if count >= t:
            count = 0
            slice.append(i)

    slice.append(len(fixations))
    return slice


def video_generator(outname, slice, input_addr, fps=1, size=(1920, 1080)):
    """Convert images into a video and save it.

    arguments

    outname	-	name of generated video
    slice	-	slice location of input images
    input_addr	-	address of input images
    fps		-	fps of generated video. Default value is 1, which means 10 seconds vidoe would
                be generated if 10 images are input.
    size	-	size of input images

    returns
    a video will be generated in the set file.
    """

    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H264
    videowrite = cv2.VideoWriter(video_addr + outname + '.mp4', fourcc, fps,
                                 size)

    for j in range(len(slice)):
        img = cv2.imread(input_addr + '/%d.jpg' % j)
        videowrite.write(img)
    cv2.destroyAllWindows()
    videowrite.release()


def fixations_integration(time, fixations, dur=1.0):
    t = dur * 1000000
    st = time[0] + t
    slice = [0]
    for i in range(len(fixations)):
        if fixations[i][0] >= st:
            slice.append(i)
            st = st + t
    slice.append(len(fixations))

    fixtion_integr = []
    for i in range(len(slice) - 1):
        bs = [0, 0, 0, 0, 0]
        bs[0] = fixations[slice[i]][0]  # start time
        bs[1] = fixations[slice[i + 1] - 1][1]  # end time

        x = 0
        y = 0
        dur = 0
        for j in range(slice[i + 1] - slice[i]):
            dur = dur + fixations[slice[i] + j][2]
            x = x + fixations[slice[i] + j][3]
            y = y + fixations[slice[i] + j][4]
        bs[2] = dur
        bs[3] = x / (slice[i + 1] - slice[i])
        bs[4] = y / (slice[i + 1] - slice[i])
        fixtion_integr.append(bs)

    return fixtion_integr


def turnback_count(fixations, threshhold=90):
    fix = parse_fixations(fixations)
    x_loc = fix['x']
    y_loc = fix['y']
    dur = fix['dur']

    turnback = []
    count = 0
    for i in range(len(x_loc) - 2):
        v1 = (x_loc[i + 1] - x_loc[i], y_loc[i + 1] - y_loc[i])
        v2 = (x_loc[i + 2] - x_loc[i + 1], y_loc[i + 2] - y_loc[i + 1])
        dx = v2[0] - v1[0]
        dy = v2[1] - v1[1]
        angle = math.atan2(dy, dx)
        angle = angle * 180 / math.pi
        turnback.append(angle)
        if angle >= threshhold:
            count = count + 1

    turnback = np.array(turnback)
    fix_dur = np.array(dur)
    return fix_dur, count, turnback


def step_count(fixations):
    fix = parse_fixations(fixations)
    x_loc = fix['x']
    y_loc = fix['y']
    dur = fix['dur']

    step_list = []
    for i in range(len(x_loc) - 1):
        p1 = np.array([x_loc[i], y_loc[i]])
        p2 = np.array([x_loc[i + 1], y_loc[i + 1]])
        p3 = p2 - p1
        step = math.hypot(p3[0], p3[1])
        step_list.append(round(step))
    step_list = np.array(step_list)
    return step_list


def find_dominate_aoi(fixations, aoi_num=6):
    fix = parse_fixations(fixations)
    x_loc = fix['x']
    y_loc = fix['y']
    dur = fix['dur']

    if aoi_num != 6:
        x_step = 1920 / 4
        y_step = 1080 / 4
    else:
        x_step = 1920 / 4
        y_step = 1080 / 2

    aoi = []
    for i in range(len(x_loc)):
        x = int((x_loc[i] // x_step) + 1)
        y = int((y_loc[i] // y_step) + 1)
        aoi.append(x + (y - 1) * 1920 / x_step)
    dominate_aoi = max(aoi, key=aoi.count)
    return int(dominate_aoi)


def find_steppulse_index(step_list, thresh=3):
    threshold = max(step_list) / thresh
    step_index = list()
    for j in range(len(step_list)):
        if step_list[j] >= threshold:
            step_index.append(j)
    return step_index

def jump_aoiLocation(step_index, fixations):
    aoi_loc = list()
    for i in range(len(step_index) - 1):
        top_left = (int(fixations[step_index[i] + 1][3]), int(fixations[step_index[i] + 1][4]), 0)
        bottom_right = (int(fixations[step_index[i + 1] ][3]), int(fixations[step_index[i + 1] ][4]), 1)
        aoi_loc.append(top_left)
        aoi_loc.append(bottom_right)
    return aoi_loc


def checkOverlap(aoi_location):
    # Reshape the structure of aoi_location into (x,y,w,h)
    aoi = list()
    for i in range(0, len(aoi_location), 2):
        x1, y1, z1 = aoi_location[i]
        x2, y2, z2 = aoi_location[i + 1]
        x = min(x1, x2)
        y = min(y1, y2)
        w = max(x1, x2) - min(x1, x2)
        h = max(y1, y2) - min(y1, y2)
        aoi.append((x, y, w, h))
    # aoi is the aoi_location with new structure
    # check if there is an overlap between different bounding boxes
    state = [0] * len(aoi)
    for i in range(len(aoi)):
        X1, Y1, W1, H1 = aoi[i]
        for j in range(i + 1, len(aoi)):
            X2, Y2, W2, H2 = aoi[j]
            colOverlap = min(X1 + W1, X2 + W2) - max(X1, X2)
            rowOverlap = min(Y1 + H1, Y2 + H2) - max(Y1, Y2)
            if colOverlap <= 0 or rowOverlap <= 0:
                continue
            elif colOverlap and rowOverlap > 0:
                if state[i] == 0:
                    state[i] = (max(state) + 1)
                    state[j] = state[i]
                else:
                    state[j] = state[i]
    return state

def remove_overlap(aoi_location, state):
    aoi = list()
    for i in range(max(state)+1):
        times = state.count(i)
        start = 0
        stop = len(state)
        loc = []
        for j in range(times):
            if start < stop:
                start = state.index(i,start,stop)
                loc.append(start)
                start = start+1
        if i == 0:
            for k in range(len(loc)):
                aoi.append(aoi_location[loc[k]*2])
                aoi.append(aoi_location[loc[k]*2+1])
        else:
            minx, miny, minz = aoi_location[loc[0]]
            maxx, maxy, maxz = aoi_location[loc[0]+1]
            for z in range(len(loc)):
                x1, y1, z1 = aoi_location[loc[z]*2]
                x2, y2, z2 = aoi_location[loc[z]*2+1]
                minx = min(minx, x1, x2)
                miny = min(miny, y1, y2)
                maxx = max(maxx, x1, x2)
                maxy = max(maxy, y1, y2)
            aoi.append((minx,miny,0))
            aoi.append((maxx,maxy,1))
    return aoi

def data_clean(data_path, set_path):
    data_msg = pd.read_csv(set_path, sep='\t', header=0)
    sys_time = np.array(data_msg['system_time_stamp'])
    msg = np.array(data_msg['msg'])
    image_info = []
    for i in range(len(msg)):
        if msg[i] == 'fix off':
            for j in range(i + 1, len(msg), 2):
                image_info.append(msg[j])
                image_info.append(sys_time[j])
                image_info.append(sys_time[j + 1])
    gazedata = pd.read_csv(data_path)
    sys_time = np.array(gazedata['system_time_stamp'])
    index = []
    for i in range(0, len(image_info), 3):
        st = image_info[i + 1]
        et = image_info[i + 2]
        j = 0
        while True:
            if sys_time[j] >= st:
                index.append(j)
                break
            j += 1
        i = 0
        while True:
            if sys_time[i] >= et:
                index.append(i - 1)
                break
            if i == (len(sys_time) - 1):
                index.append(i)
                break
            i += 1

    x_l = np.array(gazedata['left_gaze_point_on_display_area_x'])
    x_r = np.array(gazedata['right_gaze_point_on_display_area_x'])
    y_l = np.array(gazedata['left_gaze_point_on_display_area_y'])
    y_r = np.array(gazedata['right_gaze_point_on_display_area_y'])
    for i in range(0, len(index), 2):
        time = sys_time[index[i]:index[i + 1] + 1]
        x_1 = x_l[index[i]:index[i + 1] + 1] * 1920
        x_2 = x_r[index[i]:index[i + 1] + 1] * 1920
        y_1 = y_l[index[i]:index[i + 1] + 1] * 1080
        y_2 = y_r[index[i]:index[i + 1] + 1] * 1080
        df = pd.DataFrame({'time_stamp': time,
                           'x_cod_left_gaze': x_1,
                           'y_cod_left_gaze': y_1,
                           'x_cod_right_gaze': x_2,
                           'y_cod_right_gaze': y_2
                           })
        df.to_csv(globalVal.dataset_path+'dataset%d.csv' %(i/2))
    image_csv = []
    for i in range(0, len(image_info), 3):
        image_csv.append(image_info[i][-8:-5])
    df = pd.DataFrame(image_csv)
    df.to_csv(globalVal.dataset_path+'image.csv')



