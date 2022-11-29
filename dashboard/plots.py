import os

import PIL.Image as Image
import cv2
import matplotlib
import numpy
import numpy as np
from matplotlib import pyplot, image
from matplotlib.backends.backend_agg import FigureCanvasAgg
import plotly.graph_objects as go

from util import parse_fixations

# Ploting function based on Pygaze 

# # # # #
# LOOK

# COLOURS
# all colours are from the Tango colourmap, see:
# http://tango.freedesktop.org/Tango_Icon_Theme_Guidelines#Color_Palette
COLS = {"butter": ['#fce94f',
				   '#edd400',
				   '#c4a000'],
		"orange": ['#fcaf3e',
				   '#f57900',
				   '#ce5c00'],
		"chocolate": ['#e9b96e',
					  '#c17d11',
					  '#8f5902'],
		"chameleon": ['#8ae234',
					  '#73d216',
					  '#4e9a06'],
		"skyblue": ['#729fcf',
					'#3465a4',
					'#204a87'],
		"plum": ['#ad7fa8',
				 '#75507b',
				 '#5c3566'],
		"scarletred": ['#ef2929',
					   '#cc0000',
					   '#a40000'],
		"aluminium": ['#eeeeec',
					  '#d3d7cf',
					  '#babdb6',
					  '#888a85',
					  '#555753',
					  '#2e3436'],
		}
# FONT
FONT = {'family': 'Ubuntu',
		'size': 12}
matplotlib.rc('font', **FONT)


# ------------ Function to set the display size on the screen ---------------------

def draw_display(dispsize, imagefile=None):
	"""Returns a matplotlib.pyplot Figure and its axes, with a size of
	dispsize, a black background colour, and optionally with an image drawn
	onto it

	arguments

	dispsize		-	tuple or list indicating the size of the display,
					e.g. dispsize = [1920, 1080]

	keyword arguments

	imagefile		-	full path to an image file over which the heatmap
					is to be laid, or None for no image; NOTE: the image
					may be smaller than the display size, the function
					assumes that the image was presented at the centre of
					the display (default = None)

	returns
	fig, ax		-	matplotlib.pyplot Figure and its axes: field of zeros
					with a size of dispsize, and an image drawn onto it
					if an imagefile was passed
	"""
	# construct screen (black background)
	_, ext = os.path.splitext(imagefile)
	ext = ext.lower()
	data_type = 'float32' if ext == '.png' else 'uint8'
	screen = numpy.zeros((dispsize[1], dispsize[0], 3), dtype=data_type)
	# last field should be 4 for .png and 3 for jpeg
	# if an image location has been passed, draw the image
	if imagefile != None:
		# check if the path to the image exists
		if not os.path.isfile(imagefile):
			raise Exception("ERROR in draw_display: imagefile not found at '%s'" % imagefile)
		# load image
		img = image.imread(imagefile)
		# flip image over the horizontal axis
		# (do not do so on Windows, as the image appears to be loaded with
		# the correct side up there; what's up with that? :/)
		#		if not os.name == 'nt':
		#			img = numpy.flipud(img)
		# width and height of the image
		w, h = len(img[0]), len(img)
		w = int(w)
		h = int(h)
		w, h = len(img[0]), len(img)

		# x and y position of the image on the display

		x = dispsize[0] / 2 - w / 2
		y = dispsize[1] / 2 - h / 2
		x = int(x)
		y = int(y)

		# draw the image on the screen
		screen[y:y + h, x:x + w, :] += img
	# dots per inch
	dpi = 100.0
	# determine the figure size in inches
	figsize = (dispsize[0] / dpi, dispsize[1] / dpi)
	# create a figure
	fig = pyplot.figure(figsize=figsize, dpi=dpi, frameon=False)
	ax = pyplot.Axes(fig, [0, 0, 1, 1])
	ax.set_axis_off()
	fig.add_axes(ax)
	pyplot.close()
	# plot display
	ax.axis([0, dispsize[0], 0, dispsize[1]])
	ax.imshow(screen)  # , origin='upper')
	return fig, ax


# ---------- Draw raw data points on each slide ----------------------------


def draw_raw(x, y, dispsize, imagefile=None, savefilename=None):
	"""Draws the raw x and y data

	arguments

	x			-	a list of x coordinates of all samples that are to
					be plotted
	y			-	a list of y coordinates of all samples that are to
					be plotted
	dispsize		-	tuple or list indicating the size of the display,
					e.g. (1024,768)

	keyword arguments

	imagefile		-	full path to an image file over which the heatmap
					is to be laid, or None for no image; NOTE: the image
					may be smaller than the display size, the function
					assumes that the image was presented at the centre of
					the display (default = None)
	savefilename	-	full path to the file in which the heatmap should be
					saved, or None to not save the file (default = None)

	returns

	fig			-	a matplotlib.pyplot Figure instance, containing the
					fixations
	"""

	# image
	fig, ax = draw_display(dispsize, imagefile=imagefile)

	# plot raw data points
	ax.plot(x, y, 'o', color=COLS['plum'][0], markeredgecolor=COLS['plum'][2])

	# invert the y axis, as (0,0) is top left on a display
	ax.invert_yaxis()
	# save the figure if a file name was provided
	if savefilename != None:
		fig.savefig(savefilename)

	return fig


def draw_fixations(fixations, dispsize, imagefile=None, durationsize=True, durationcolour=True, alpha=0.5,
				   savefilename=None, size=300):
	"""Draws circles on the fixation locations, optionally on top of an image,
	with optional weigthing of the duration for circle size and colour

	arguments

	fixations		-	a list of fixation ending events from a single trial,
					as produced by edfreader.read_edf, e.g.
					edfdata[trialnr]['events']['Efix']
	dispsize		-	tuple or list indicating the size of the display,
					e.g. (1024,768)

	keyword arguments

	imagefile		-	full path to an image file over which the heatmap
					is to be laid, or None for no image; NOTE: the image
					may be smaller than the display size, the function
					assumes that the image was presented at the centre of
					the display (default = None)
	durationsize	-	Boolean indicating whether the fixation duration is
					to be taken into account as a weight for the circle
					size; longer duration = bigger (default = True)
	size			-   Variable to control the size of drawn circles;
					default is 300; The bigger 'size' is, the smaller circles will be drawn
	durationcolour	-	Boolean indicating whether the fixation duration is
					to be taken into account as a weight for the circle
					colour; longer duration = hotter (default = True)
	alpha		-	float between 0 and 1, indicating the transparancy of
					the heatmap, where 0 is completely transparant and 1
					is completely untransparant (default = 0.5)
	savefilename	-	full path to the file in which the heatmap should be
					saved, or None to not save the file (default = None)

	returns

	fig			-	a matplotlib.pyplot Figure instance, containing the
					fixations
	"""

	# Draw the FIXATION point on each slide --------------------------

	fix = parse_fixations(fixations)

	# IMAGE
	fig, ax = draw_display(dispsize, imagefile=imagefile)

	# CIRCLES
	# duration weigths
	if durationsize:
		siz = 1 * (fix['dur'] / size)
	else:
		siz = 1 * numpy.median(fix['dur'] / size)
	if durationcolour:
		col = fix['dur']
	else:
		col = COLS['chameleon'][2]
	# draw circles
	ax.scatter(fix['x'], fix['y'], s=siz, c=col, marker='o', cmap='jet', alpha=alpha, edgecolors='none')

	# FINISH PLOT
	# invert the y axis, as (0,0) is top left on a display
	ax.invert_yaxis()
	# save the figure if a file name was provided
	if savefilename != None:
		fig.savefig(savefilename)

	return fig


def draw_scanpath(fixations, saccades, dispsize, imagefile=None, alpha=0.5, savefilename=None):
	"""Draws a scanpath: a series of arrows between numbered fixations,
	optionally drawn over an image
	arguments

	fixations		-	a list of fixation ending events from a single trial,
					as produced by edfreader.read_edf, e.g.
					edfdata[trialnr]['events']['Efix']
	saccades		-	a list of saccade ending events from a single trial,
					as produced by edfreader.read_edf, e.g.
					edfdata[trialnr]['events']['Esac']
	dispsize		-	tuple or list indicating the size of the display,
					e.g. (1024,768)

	keyword arguments

	imagefile		-	full path to an image file over which the heatmap
					is to be laid, or None for no image; NOTE: the image
					may be smaller than the display size, the function
					assumes that the image was presented at the centre of
					the display (default = None)
	alpha		-	float between 0 and 1, indicating the transparancy of
					the heatmap, where 0 is completely transparant and 1
					is completely untransparant (default = 0.5)
	savefilename	-	full path to the file in which the heatmap should be
					saved, or None to not save the file (default = None)

	returns

	fig			-	a matplotlib.pyplot Figure instance, containing the
					heatmap
	"""

	# image
	fig, ax = draw_display(dispsize, imagefile=imagefile)

	# FIXATIONS
	# parse fixations
	fix = parse_fixations(fixations)
	# draw fixations
	ax.scatter(fix['x'], fix['y'], s=(1 * fix['dur'] / 200), c=COLS['plum'][2], marker='o', cmap='jet', alpha=alpha,
			   edgecolors='none')
	# draw annotations (fixation numbers)
	for i in range(len(fixations)):
		ax.annotate(str(i + 1), (fix['x'][i], fix['y'][i]), color=COLS['aluminium'][5], alpha=1,
					horizontalalignment='center', verticalalignment='center', multialignment='center')
	# ax.arrow(fix['x'][i],fix['y'][i], fix['x'][i+1]-fix['x'][i], fix['y'][i+1]-fix['y'][i], alpha=alpha, fc=COLS['aluminium'][0], ec=COLS['aluminium'][5],
	# fill=True, shape='full', width=3, head_width=20, head_starts_at_zero=False, overhang=0)
	# SACCADES
	if saccades:
		# loop through all saccades
		# for st, et, dur, sx, sy, ex, ey in saccades:
		for sx, sy, ex, ey in saccades:
			# draw an arrow between every saccade start and ending
			ax.arrow(sx, sy, ex - sx, ey - sy, alpha=alpha, fc=COLS['aluminium'][0], ec=COLS['aluminium'][5], fill=True,
					 shape='full', width=10, head_width=20, head_starts_at_zero=False, overhang=0)

	# invert the y axis, as (0,0) is top left on a display
	ax.invert_yaxis()
	# save the figure if a file name was provided
	if savefilename != None:
		fig.savefig(savefilename)

	return fig


def draw_heatmap(fixations, dispsize, imagefile=None, durationweight=True, alpha=0.5, savefilename=None):
	"""Draws a heatmap of the provided fixations, optionally drawn over an
	image, and optionally allocating more weight to fixations with a higher
	duration.

	arguments

	fixations		-	a list of fixation ending events from a single trial,
					as produced by edfreader.read_edf, e.g.
					edfdata[trialnr]['events']['Efix']
	dispsize		-	tuple or list indicating the size of the display,
					e.g. (1024,768)

	keyword arguments

	imagefile		-	full path to an image file over which the heatmap
					is to be laid, or None for no image; NOTE: the image
					may be smaller than the display size, the function
					assumes that the image was presented at the centre of
					the display (default = None)
	durationweight	-	Boolean indicating whether the fixation duration is
					to be taken into account as a weight for the heatmap
					intensity; longer duration = hotter (default = True)
	alpha		-	float between 0 and 1, indicating the transparancy of
					the heatmap, where 0 is completely transparant and 1
					is completely untransparant (default = 0.5)
	savefilename	-	full path to the file in which the heatmap should be
					saved, or None to not save the file (default = None)

	returns

	fig			-	a matplotlib.pyplot Figure instance, containing the
					heatmap
	"""

	# FIXATIONS
	fix = parse_fixations(fixations)

	# IMAGE
	fig, ax = draw_display(dispsize, imagefile=imagefile)

	# HEATMAP
	# Gaussian
	gwh = 200
	gsdwh = gwh / 6
	gaus = gaussian(gwh, gsdwh)
	# matrix of zeroes
	strt = gwh / 2
	heatmapsize = dispsize[1] + 2 * strt, dispsize[0] + 2 * strt
	##heatmap = numpy.zeros(heatmapsize, dtype=float)
	heatmap = numpy.zeros((1280, 2120), dtype=float)  # there was an error in code
	# create heatmap
	for i in range(0, len(fix['dur'])):
		# get x and y coordinates
		# x and y - indexes of heatmap array. must be integers
		x = strt + int(fix['x'][i]) - int(gwh / 2)
		# x = strt + fix['x'][i] - gwh/2
		y = strt + int(fix['y'][i]) - gwh / 2
		# correct Gaussian size if either coordinate falls outside of
		# display boundaries
		x = int(x)
		y = int(y)
		if (not 0 < x < dispsize[0]) or (not 0 < y < dispsize[1]):
			hadj = [0, gwh];
			vadj = [0, gwh]
			if 0 > x:
				hadj[0] = abs(x)
				x = 0
			elif dispsize[0] < x:
				hadj[1] = gwh - int(x - dispsize[0])
			if 0 > y:
				vadj[0] = abs(y)
				y = 0
			elif dispsize[1] < y:
				vadj[1] = gwh - int(y - dispsize[1])
			# add adjusted Gaussian to the current heatmap
			try:
				heatmap[y:y + vadj[1], x:x + hadj[1]] += gaus[vadj[0]:vadj[1], hadj[0]:hadj[1]] * fix['dur'][i]
			except:
				# fixation was probably outside of display
				pass
		else:
			# add Gaussian to the current heatmap
			heatmap[y:y + gwh, x:x + gwh] += gaus * fix['dur'][i]
	strt = int(strt)
	# resize heatmap
	heatmap = heatmap[strt:dispsize[1] + strt, strt:dispsize[0] + strt]
	# remove zeros
	lowbound = numpy.mean(heatmap[heatmap > 0])
	heatmap[heatmap < lowbound] = numpy.NaN
	# draw heatmap on top of image
	ax.imshow(heatmap, cmap='jet', alpha=alpha)

	# FINISH PLOT
	# invert the y axis, as (0,0) is top left on a display
	ax.invert_yaxis()
	# save the figure if a file name was provided
	if savefilename != None:
		fig.savefig(savefilename)

	return fig


def draw_arrows(fixations, index, imagefile):
	img = cv2.imread(imagefile)
	if index >= 1:
		for i in range(index):
			st = (int(fixations[i][3]), int(fixations[i][4]))
			ed = (int(fixations[i + 1][3]), int(fixations[i + 1][4]))
			cv2.arrowedLine(img, st, ed, (0, 0, 255))
	cv2.imwrite(imagefile, img)


def draw_boundingbox(aoi, imagefile, fix_number):
	img = cv2.imread(imagefile)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	for i in range(0, len(aoi), 2):
		inward = aoi[i][2] - aoi[i + 1][2]
		if inward == -1:
			top_left = (aoi[i][0], aoi[i][1])
			bottom_right = (aoi[i + 1][0], aoi[i + 1][1])
			cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 4)
		# cv2.putText(img,  'AOI: %d' % (int(i/2)+1) +' Number of fixations: %d' %fix_number[int(i/2)], (top_left[0],bottom_right[1]),
		# cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 4)

		else:
			top_left = (0, 0)
			bottom_right = (1920, 1080)
			cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 4)
			cv2.putText(img, 'AOI: %d' % (int(i / 2) + 1) + ' Number of fixations: %d' % fix_number[int(i / 2)],
						(top_left[0], bottom_right[1]),
						cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)
	return img


def draw_aoi_map(img, period):
	color = (0, 255, 0)
	thickness = 4
	if period == 10:
		pt1 = [(0, 1080), (0, 540), (0, 0), (480, 0), (960,0), (1440, 0), (1920, 0)]
		pt2 = [(1920, 1080), (1920, 540), (1920, 0), (480, 1080), (960,1080), (1440, 1080), (1920, 1080)]
		text_loc = [(160, 270), (640, 270), (1120, 270), (1600, 270), (160, 810), (640, 810), (1120, 810), (1600, 810)]
	elif period == 5:
		pt1 = [(0, 1080),(0, 810), (0, 540), (0, 270), (0, 0), (480, 0), (960, 0), (1440, 0), (1920, 0)]
		pt2 = [(1920, 1080), (1920, 810), (1920, 540), (1920, 270), (1920, 0), (480, 1080), (960, 1080), (1440, 1080), (1920, 1080)]
		text_loc = [(160, 135), (640, 135), (1120, 135), (1600, 135),
					(160, 405), (640, 405), (1120, 405), (1600, 405),
					(160, 675), (640, 675), (1120, 675), (1600, 675),
					(160, 945), (640, 945), (1120, 945), (1600, 945),]
	for i in range(len(pt1)):
		cv2.line(img, pt1[i], pt2[i], color, thickness)
	for j in range(len(text_loc)):
		cv2.putText(img, 'AOI %d' % (j + 1), text_loc[j],
					cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 4)

def draw_guage(ConcenLevel = 300):
	fig = go.Figure(go.Indicator(
		mode="gauge+number",
		value=ConcenLevel,
		domain={'x': [0, 1], 'y': [0, 1]},
		# title = {'text': "Level of Concentration", 'font': {'size': 24}},
		gauge={
			'axis': {'range': [None, 500], 'tickwidth': 1, 'tickcolor': "blue"},
			'bar': {'color': 'rgb(255,215,0)'},
			'bgcolor': "white",
			'borderwidth': 2,
			'bordercolor': "gray",
			'steps': [
				{'range': [0, 100], 'color': 'rgb(240,248,255)'},
				{'range': [100, 200], 'color': 'rgb(176,196,222)'},
				{'range': [200, 300], 'color': 'rgb(100,149,237)'},
				{'range': [300, 400], 'color': 'rgb(65,105,225)'},
				{'range': [400, 500], 'color': 'rgb(0,0,205)'}],
			'threshold': {
				'line': {'color': 'rgb(255,215,0)', 'width': 4},
				'thickness': 0.75,
				'value': 500}}))
	fig.update_layout(paper_bgcolor="white", font={'color': 'rgb(85,146,239)', 'family': "Arial"})
	return fig.to_image()
# ----------------- Helper functions ---------------

def gaussian(x, sx, y=None, sy=None):
	"""Returns an array of numpy arrays (a matrix) containing values between
	1 and 0 in a 2D Gaussian distribution

	arguments
	x		-- width in pixels
	sx		-- width standard deviation

	keyword argments
	y		-- height in pixels (default = x)
	sy		-- height standard deviation (default = sx)
	"""

	# square Gaussian if only x values are passed
	if y == None:
		y = x
	if sy == None:
		sy = sx
	# centers
	xo = x / 2
	yo = y / 2
	# matrix of zeros
	M = numpy.zeros([y, x], dtype=float)
	# gaussian matrix
	for i in range(x):
		for j in range(y):
			M[j, i] = numpy.exp(
				-1.0 * (((float(i) - xo) ** 2 / (2 * sx * sx)) + ((float(j) - yo) ** 2 / (2 * sy * sy))))

	return M

def image_convert(image):
	"""Returns an array of numpy arrays

	arguments
	image	-- a di instance
	"""

	canvas = FigureCanvasAgg(image)
	canvas.draw()
	w, h = canvas.get_width_height()
	buf = np.fromstring(canvas.tostring_argb(), dtype=np.uint8)
	buf.shape = (w, h, 4)
	buf = np.roll(buf, 3, axis=2)
	fig = Image.frombytes("RGBA", (w, h), buf.tostring())
	fig = np.asarray(fig)

	return fig



