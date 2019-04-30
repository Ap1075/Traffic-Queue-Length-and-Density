import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", metavar="--path", dest="vid_path",
	                help="String containing video file's path", type=str, required = True)
args = parser.parse_args()
cap = cv2.VideoCapture(args.vid_path)
frame_array = []
old_frame = None
frame_no=0
old_frame_no=0
# fpbg=cv2.bgsegm.createBackgroundSubtractorMOG()
out = cv2.VideoWriter("./corners15.avi", cv2.VideoWriter_fourcc(*'DIVX'), 25, (1920,1080))
mask = cv2.imread("./l2r.jpg",0)
(tr, mask) = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# print("otsu thresh", tr)
mask = cv2.bitwise_not(mask)
# print("xx shape:", mask.shape)
mask = cv2.medianBlur(mask, 51)

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2

cv2.imshow("saved l2r", mask)
while True:
	_, frame = cap.read()
	frame_no += 1
	# print("frame number",frame_no)
	# if frame_no % 5 ==0 and frame_no <50:
		# cv2.imwrite("./out{}.jpg".format(frame_no),frame)
	gr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	gabored = cv2.getGaborKernel((17,17), 10.0, np.pi/6, 6.7, 0.75, 5, ktype=cv2.CV_32F)
	filtered_img = cv2.filter2D(frame, cv2.CV_8UC3, gabored)
	gr_fil = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2GRAY)

	
	# cv2.imshow("mask l2r", mask)
	# kernel = np.ones((15,15),np.float32)/225
	# mask = cv2.filter2D(mask,-1,kernel)

	thresh2 = cv2.adaptiveThreshold(gr_fil,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)         #thresh 2 is grayscaled filter.
	# thresh2 = cv2.threshold(gr_fil,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	# print(thresh2.shape)
	med = cv2.medianBlur(thresh2, 101)
	# blur = cv2.GaussianBlur(thresh2, (31,31), 0)														### blur
	thresh3 = cv2.bitwise_not(med)
	kernel = np.ones((9,9),np.float32)/81
	thresh3 = cv2.filter2D(thresh3,-1,kernel)
	# thresh3 = cv2.fastNlMeansDenoisingColored(thresh3,None,10,10,7,21)
	# new_fr = cv2.bitwise_and(frame, frame, mask = thresh3)
	new_fr = cv2.bitwise_and(frame, frame, mask = mask)
	# new_fr = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2GRAY)
	# ret2,new_fr = cv2.threshold(new_fr,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	inv =  cv2.bitwise_not(filtered_img)
	#fgmask = fpbg.apply(gr)
	greyed = cv2.cvtColor(new_fr, cv2.COLOR_BGR2GRAY)
	corns = cv2.goodFeaturesToTrack(greyed, 50, 0.5, 7)
	corns = np.int0(corns)
	corn_neigh=[]						# contains windows for each corner in the frame (theres a win(dow) for each corner)
	for i,corner in enumerate(corns):
		x,y = corner.ravel()
		# cv2.circle(frame, (x,y), 10,255,-1)	
		# corn_neigh.append([y-1,y+2, x-1,x+2])  # taking 7x7 neighbourhood around the corner pixel
		corn_neigh.append([y-1,y+1, x-1,x+1])  # taking 7x7 neighbourhood around the corner pixel

	if old_frame is not None:
		old_frame_no += 1
		print("current frame", frame_no)
		print("old_frame", old_frame_no)
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
		print(new_fr.shape)

		for corn in stat:
			cv2.circle(frame, (corn[1],corn[0]), 10,255,-1)	        # blue stat
			cv2.circle(new_fr, (corn[1],corn[0]), 10,255,-1)
		
		for corn in dyn:
			cv2.circle(frame, (corn[1],corn[0]), 10,(0,0,255),-1)	# red dyn

		cv2.putText(frame,'Static Corners: {0}, Dynamic Corners: {1}'.format(len(stat), len(dyn)), 
				    bottomLeftCornerOfText, 
				    font, 
				    fontScale,
				    fontColor,
				    lineType)
		
		out.write(frame)

			# max_val.append(np.max(b))

	frame_array.append(frame)
	old_frame = gr
	# cv2.imshow("fgmask",fgmask)
	# cv2.imshow("inv_img",inv)
	# cv2.imshow("filtered_img",filtered_img)
	cv2.imshow("thresh3",thresh3)
	cv2.imshow("blurred median",med)
	cv2.imshow("frame",frame)
	# cv2.imshow("applied tgresh3",new_fr)
	# cv2.imshow("fft",magnitude_spectrum)
	k = cv2.waitKey(1)
	if k == ord("q"):
		cv2.destroyAllWindows()
		break
	elif k == ord("s"):
		cv2.imwrite("./saver_outs/best_mask.jpg", thresh3)
	else:
		pass
	cv2.waitKey(1)
# print(max_val)

out.release()
cap.release()