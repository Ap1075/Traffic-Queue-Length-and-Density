import cv2
import numpy as np

cap = cv2.VideoCapture("/home/armaan/Videos/videofiles_bridge/8Mar_1pm.mp4")
kp = 0
mask = cv2.imread("./l2r.jpg",0)
kernel = np.ones((9,9),np.float32)/81
mask = cv2.filter2D(mask,-1,kernel)
# mask = cv2.adaptiveThreshold(mask,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,2)
while True:
	_, frame = cap.read()
	# new_fr = cv2.bitwise_and(frame, frame, mask = mask)
	cv2.imshow('frame',frame)
	# cv2.imshow('newer frame',mask)
	# cv2.imshow('new frame',new_fr)
	# print(frame.shape)
	# print((frame.shape[0]**2 + frame.shape[1]**2)**(1/2))
	k = cv2.waitKey(20)
	if k == ord("s"):
		kp+=1
		print("saving", kp)
		cv2.imwrite("saver_outs/new_img_foot{0}.jpg".format(kp), frame)
	elif k == ord("q"):
		print("quitting")
		break
	else:
		pass
cv2.destroyAllWindows()
cap.release()