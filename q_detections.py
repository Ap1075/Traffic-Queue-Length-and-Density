import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
from scipy import optimize
from operator import itemgetter
import math

parser = argparse.ArgumentParser()
parser.add_argument("-p", metavar="--path", dest="vid_path",
	                help="String containing video file's path", type=str, required = True)
parser.add_argument("-s", metavar="--out_path", dest="out_path",
	                help="String containing output video file's path", type=str, required = True)
args = parser.parse_args()
cap = cv2.VideoCapture(args.vid_path)
old_frame = None
frame_no=0
old_frame_no=0
queue_ind = False
stat_greater_count = 0
dyn_greater_count = 0
old_q=[]
sd=0
medd=0
out = cv2.VideoWriter(args.out_path, cv2.VideoWriter_fourcc(*'DIVX'), 25, (1920,1080))
mask = cv2.imread("masks_gate/mafg.png",0)
mask = cv2.medianBlur(mask, 51)#####################################################

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (1000,850)
bottomRightCornerOfText = (1000,870)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2

def q_measure(M, src, dst):
	xnew1, ynew1,znew1 = M.dot(np.array(src)) 		# src and dst must be of type (x,y,1)
	xnew2, ynew2,znew2 = M.dot(np.array(dst))
	distance = math.sqrt( ((xnew1-xnew2)**2)+((ynew1-ynew2)**2) +((znew1-znew2)**2) )
	return distance

cv2.imshow("saved l2r", mask)
while True:
	_, frame = cap.read()
	frame_no += 1
	inpt_array = [[1421,895],[1454,988],[1149,618],[1116,592],[772,785],[838,735]] # gratings 1 and 2, sedan1 , ace , sedan 2 rear and front
	dst_array = [[0.45, 15.8],[0,17],[4.5,2.2],[7.7,0],[7.5,12.4],[7.5,8.8]]
	if frame_no == 2:
		M,mask_ho = cv2.findHomography(np.array(inpt_array), np.array(dst_array),cv2.LMEDS,5)
		print("H matrix:\n", M)
		scale = 2.5 																					# experimental param
		M = scale*M
		xnew1, ynew1,znew1 = M.dot(np.array((39,1078,1)))
		xnew2, ynew2,znew2 = M.dot(np.array((1015,580,1)))
		print("\n")
		distance = math.sqrt( ((xnew1-xnew2)**2)+((ynew1-ynew2)**2) +((znew1-znew2)**2) )
		# print("distance (in metres):",distance)

	gr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	new_fr = cv2.bitwise_and(frame, frame, mask = mask)

	greyed = cv2.cvtColor(new_fr, cv2.COLOR_BGR2GRAY)
	cv2.imshow("inspect this", greyed)
	corns = cv2.goodFeaturesToTrack(greyed, 100, 0.5, 10)
	corns = np.int0(corns)
	corn_neigh=[]						# contains windows for each corner in the frame (theres a win(dow) for each corner)
	for i,corner in enumerate(corns):
		x,y = corner.ravel()
		corn_neigh.append([y-1,y+1, x-1,x+1])  # taking 7x7 neighbourhood around the corner pixel

	if old_frame is not None:
		old_frame_no += 1
		max_val = []             # list containing max val for each window around a corner for consecutive frames.
		stat = []
		dyn = []
		for i,win in enumerate(corn_neigh):
			max_val.append(np.max(np.subtract(gr[win[0]:win[1], win[2]:win[3]], old_frame[win[0]:win[1], win[2]:win[3]])))
			thresh = 0.5*(np.max(gr[win[0]:win[1], win[2]:win[3]]) - np.min(gr[win[0]:win[1], win[2]:win[3]]))
			# print("threshold", thresh)
			# if win[0]>260 and win[2]<1450:
			if win[0]>350 and win[2]<1400:
				stat.append([win[0]+1, win[2]+1]) if max_val[i]<=thresh else dyn.append([win[0]+1, win[2]+1])
		
		if ((len(stat) > 10) or (len(stat)>len(dyn) and len(dyn)<5)):						## after false positives are gone, a simpler if.
			stat_greater_count +=1
			if stat_greater_count > 10:
				queue_ind = True
				stat = sorted(stat, key=itemgetter(1))
				if len(stat) == 1:
					continue
				cv2.circle(frame, tuple((stat[-2][1], stat[-2][0])), 15,(0,255,0),-1)
				cv2.circle(frame, tuple((stat[0][1], stat[0][0])), 15,(0,255,0),-1)
				medd = round(q_measure(M, [stat[0][1], stat[0][0],1], [stat[-2][1], stat[-2][0],1]), 3)
				if len(old_q)>2 and  (medd < np.mean(old_q)-2*np.std(old_q)):
					medd = round(np.mean(old_q),2)
				old_q.append(medd)
				if len(old_q)>2:
					# print("queue", medd)
				dyn_greater_count = 0
				stat_greater_count = 0

		elif len(dyn) > 1.5*len(stat) or len(stat)<10:
			dyn_greater_count +=1
			# sd =0
			if dyn_greater_count > 15:
				queue_ind = False
				# old_q = []
				stat_greater_count = 0
				dyn_greater_count = 0
			# print(stat_greater_count)
		if queue_ind:			
			cv2.putText(frame,"Queue Length: {0} metres".format(medd), (1000,800), font, 1.5, (255,0,255), 3)
			if len(old_q)>3:
			# 	medd = math.ceil(np.median(old_q))
			# 	# if len(old_q) > 10 or np.std(old_q):
				if len(old_q)>10 or (max(old_q) - min(old_q)>10):
					old_q = []

		for corn in stat:
			# if corn[0] > 280:
				# if corn[0] < 2*std(corn[0]):
			cv2.circle(frame, (corn[1],corn[0]), 7,(255,0,0),-1)	        # blue stat
			cv2.circle(new_fr, (corn[1],corn[0]), 7,(255,0,0),-1)
				# cv2.circle(frame, (200,500), 100,255,-1)
				# cv2.putText(frame,'{0}, {1}'.format(corn[1], corn[0]), (corn[1],corn[0]), font, 0.5, (255,0,255), 2)		
		for corn in dyn:
			cv2.circle(frame, (corn[1],corn[0]), 7,(0,0,255),-1)	# red dyn
			# cv2.putText(frame,'{0}, {1}'.format(corn[1], corn[0]), (corn[1],corn[0]), font, 0.5, (255,0,255), 2)

		cv2.putText(frame,'Static Corners: {0}, Dynamic Corners: {1}'.format(len(stat), len(dyn)), 
				    bottomLeftCornerOfText, 
				    font, 
				    fontScale,
				    fontColor,
				    lineType)

	out.write(frame)

			# max_val.append(np.max(b))

	# frame_array.append(frame)
	old_frame = gr

	cv2.imshow("frame",frame)
	k = cv2.waitKey(1)
	if k == ord("q"):
		cv2.destroyAllWindows()
		break
	elif k == ord("s"):
		cv2.imwrite("./saver_outs/best_mask.jpg", frame)
	else:
		pass
	cv2.waitKey(1)
out.release()
cap.release()