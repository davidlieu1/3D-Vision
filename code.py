import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.spatial import Delaunay
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import cv2
import trimesh
import math
from visutils import *
import selectpoints
from camutils import *
from meshutils import *
import pickle

def calib():
	import calibrate
	
	open_file = open('calibration.pickle', 'rb')
	intr = pickle.load(open_file) # calibration data
	open_file.close()
	
	b = 5 #btwn cams
	d = 10 # dist
	tL = np.array([[-(b/2),0,0]]).T
	tR = np.array([[(b/2),0,0]]).T
	p_init = np.array([0,0,0,0,0, -2])
	f = (intr['fx']+intr['fy'])/2 # focal length
	c = np.array([[intr['cx']], [intr['cy']]]) # principle point

	camR = Camera(f,c, makerotation(0,0,0), tR)
	camL = Camera(f,c, makerotation(0,0,0), tL)
	imgL = plt.imread('calib_jpg_u/frame_C1_01.jpg')
	
	ret, cornersL = cv2.findChessboardCorners(imgL, (8,6), None)
	pts2L = cornersL.squeeze().T
	imgR = plt.imread('calib_jpg_u/frame_C0_01.jpg')
	ret, cornersR = cv2.findChessboardCorners(imgR, (8,6), None)
	pts2R = cornersR.squeeze().T
	pts3 = np.zeros((3,6*8))
	yy,xx = np.meshgrid(np.arange(8),np.arange(6))
	pts3[0,:] = 2.8*xx.reshape(1,-1)
	pts3[1,:] = 2.8*yy.reshape(1,-1)
	
	camL = calibratePose(pts3, pts2L, camL,p_init)
	camR = calibratePose(pts3, pts2R, camR,p_init)
	
	return camL, camR
def color(imprefixL, imprefixR):
	# read color file
	cL = plt.imread(imprefixL+'01.png') 
	cR = plt.imread(imprefixR+'01.png') 
	l = [[],[],[]]
	r = [[],[],[]]
	# append array of each different color to create arr of shape 3xN
	for i in range(len(pts2L[0])):
		l[0].append(cL.T[0][pts2L[0][i]][pts2L[1][i]])
		l[1].append(cL.T[1][pts2L[0][i]][pts2L[1][i]])
		l[2].append(cL.T[2][pts2L[0][i]][pts2L[1][i]])
	rgbL = np.asarray(l)
	
	for i in range(len(pts2R[0])):
		r[0].append(cR.T[0][pts2R[0][i]][pts2R[1][i]])
		r[1].append(cR.T[1][pts2R[0][i]][pts2R[1][i]])
		r[2].append(cR.T[2][pts2R[0][i]][pts2R[1][i]])
	rgbR = np.asarray(r)
	
	return rgbL, rgbR
def box_pruning(pts3, boxlimits):
	used = []
	for i in range(pts3.T.shape[0]):
		if pts3.T[i][0] >= boxlimits[0] and pts3.T[i][0] <= boxlimits[1] and \
			pts3.T[i][1] >= boxlimits[2] and pts3.T[i][1] <= boxlimits[3] and \
			pts3.T[i][2] >= boxlimits[4] and pts3.T[i][2] <= boxlimits[5]:
			used.append(i)

	return used
def triangle_pruning(pts3, tri, trithresh):
	tris = pts3.T[tri.simplices]
	badtris = []
	for i in range(len(tris)):
		if np.linalg.norm(tris[i][0]-tris[i][1]) > trithresh or \
			np.linalg.norm(tris[i][0]-tris[i][2]) > trithresh or \
			np.linalg.norm(tris[i][1]-tris[i][2]) > trithresh:
			badtris.append(i)
			
	tri.simplices = np.delete(tri.simplices,badtris, axis = 0)

camL, camR = calib()
for i in range(6):
	imprefixL = 'couple/grab_{}_u/frame_C1_'.format(i)
	imprefixR = 'couple/grab_{}_u/frame_C0_'.format(i)
	imprefixCL = 'couple/grab_{}_u/color_C1_'.format(i)
	imprefixCR = 'couple/grab_{}_u/color_C0_'.format(i)
	threshold = 0.02
	cthresh = 0.17
	pts2L,pts2R,pts3 = reconstruct(imprefixCL, imprefixCR, imprefixL, imprefixR, threshold, cthresh, camL, camR)
	vis_scene(camL,camR,pts3,looklength=20)
	colorsL,colorsR = color(imprefixCL,imprefixCR)
	boxlimits = np.array([-30,15,-50,50,-30,-10])

	used = box_pruning(pts3, boxlimits)
	triL = Delaunay(pts2L.T[used])
	triR = Delaunay(pts2R.T[used])
	trithresh = 7
	triangle_pruning(pts3, triL, trithresh)
	triangle_pruning(pts3, triR, trithresh)

	vis_scene(camL,camR,pts3.T[used].T,looklength=20)
	writeply(pts3,colorsL,triL.simplices,'scan{}L.ply'.format(i))
	writeply(pts3,colorsR,triR.simplices,'scan{}R.ply'.format(i))