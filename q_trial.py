import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse

cap = cv2.VideoCapture("/home/armaan/Videos/videofiles_bridge/6Mar.mp4")
frame_array = []
old_frame = None
frame_no=0
old_frame_no=0
fpbg=cv2.bgsegm.createBackgroundSubtractorMOG()
out = cv2.VideoWriter("./corners.avi", cv2.VideoWriter_fourcc(*'DIVX'), 25, (1920,1080))
while True:
	_, frame = cap.read()
	frame_no += 1
	# print("frame number",frame_no)
	if frame_no % 5 ==0 and frame_no <50:
		cv2.imwrite("./out{}.jpg".format(frame_no),frame)
	gr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	gabored = cv2.getGaborKernel((17,17), 10.0, np.pi/6, 5.0, 0.75, 5, ktype=cv2.CV_32F)
	filtered_img = cv2.filter2D(frame, cv2.CV_8UC3, gabored)
	fgmask = fpbg.apply(gr)
	corns = cv2.goodFeaturesToTrack(gr, 70, 0.01, 7)
	corns = np.int0(corns)
	corn_neigh=[]						# contains windows for each corner in the frame (theres a win(dow) for each corner)
	for i,corner in enumerate(corns):
		x,y = corner.ravel()
		# cv2.circle(frame, (x,y), 10,255,-1)	
		corn_neigh.append([y-1,y+2, x-1,x+2])  # taking 7x7 neighbourhood around the corner pixel

	if old_frame is not None:
		old_frame_no += 1
		# print("old_frame_no",old_frame_no)
		# add frame differencing code here.
		max_val = []             # list containing max val for each window around a corner for consecutive frames.
		stat = []
		dyn = []
		for i,win in enumerate(corn_neigh):
			max_val.append(np.max(np.subtract(gr[win[0]:win[1], win[2]:win[3]], old_frame[win[0]:win[1], win[2]:win[3]])))
			thresh = 0.3*(np.max(gr[win[0]:win[1], win[2]:win[3]]) - np.min(gr[win[0]:win[1], win[2]:win[3]]))
			# print("threshold", thresh)
			stat.append([win[0]+1, win[2]+1]) if max_val[i]<=thresh else dyn.append([win[0]+1, win[2]+1])
		print('stat corners:',len(stat))
		print('dyn corners:',str(len(dyn))+"\n")

		for corn in stat:
			cv2.circle(frame, (corn[1],corn[0]), 10,255,-1)	        # blue stat
		
		# for corn in dyn:
		# 	cv2.circle(frame, (corn[1],corn[0]), 10,(0,0,255),-1)	# red dyn

			# out.write(frame)

			# max_val.append(np.max(b))

	frame_array.append(frame)
	old_frame = gr
	cv2.imshow("fgmask",fgmask)
	cv2.imshow("frame",frame)
	cv2.imshow("filtered_img",filtered_img)
	k = cv2.waitKey(33)
	if k == ord("q"):
		cv2.destroyAllWindows()
		break
	cv2.waitKey(100)
# print(max_val)

out.release()
cap.release()