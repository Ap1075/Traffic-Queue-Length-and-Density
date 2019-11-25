from __future__ import print_function, absolute_import
import cv2
#from deepsort import deepsort_rbc
from yolov3 import YOLOV3
import cv2
import os
import time, math
import numpy as np
from scipy.optimize import least_squares, leastsq 
from auto_track import analytics_rbc
import hourglass_single #.example.hourglass_single #import hourglass
import csv
import json

weights = np.ones((48))

def compute_homography(image_keypoints, world_keypoints):
    "Takes image points as a list of lists. 24 lists for 24 keypoints. Use 6 different auto positions (from separate frames of same view) and the origin3d point to compute homography by establishing 6 point correspondences."


def array_dump(filename, img_coords, real_coords):
	with open(filename, "w") as arraydump:
		json.dump([img_coords,real_coords], arraydump)

def reproject_error(pose, points, imagepoints): #pose (rvecs, tvecs), world_points, image_points
	
	global weights
	finetuned = get_error(pose, points)
	#finetuned_reshaped = finetuned.reshape((-1,2))
	err = np.multiply((imagepoints - finetuned), weights)
	update_weights(err)
#	print (weights)
#	time.sleep(0.1)	
	return err

def update_weights(err):

	global weights
	for i in range(len(err)):
		weights[i] = 1 / (1 + err[i])

	#weights*=100.0

def get_error(pose, points):

	pts={0: [45.0, 256.5, 71.0], 1: [-45.0, 256.5, 71.0], 2: [56.0, 186.7, 89.0], 3: [-56.0, 186.7, 89.0], 4: [51.0, 240.0, 149.5], 5: [-51.0, 240.0, 149.5], 6: [0.0, 260.0, 41.5], 7: [-47.5, 43.6, 104.7], 8: [-57.0, 57.8, 145.5], 9: [-54.3, 135.0, 149.8], 10: [-66.4, 220.0, 19.8], 11: [-43.0, 68.0, 165.6], 12: [57.0, 57.8, 145.5], 13: [54.3, 135.0, 149.8], 14: [43.0, 68.0, 165.6], 15: [47.5, 43.6, 104.7], 16: [66.4, 220.0, 19.8], 17: [44.2, 61.5, 152.4], 18: [-44.2, 61.5, 152.2], 19: [49.0, 56.0, 116.5], 20: [-49.0, 56.0, 116.5], 21: [0.0, 0.0, 38.0], 22: [33.0, 36.0, 83.5], 23: [-33.0, 36.0, 83.5]}
	#mtx = np.array([[1324.110551, 0.000000, 993.993108], [0.000000, 1324.110210, 621.997610],[  0. ,   0. ,   1. ]])
	#dist = np.array([[-0.401747, 0.148985, -0.008159, -0.006626, 0.000000]]) 
	mtx = np.array([[1515.104, 0.000000, 665.102], [0.000000, 1523.561, 342.34],[  0. ,   0. ,   1. ]])
	dist = np.array([[-0.27888534, -0.15845886, -0.00238336,  0.00136916,  1.09498805]]) 
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

	imagepoints_finetuned = np.empty((len(points)*2), dtype=np.float)

	for i in range(len(points)):
		ponting = np.array([[pts[points[i]]]], dtype=np.float)
		imgpoints2, _ = cv2.projectPoints(ponting, pose[0:3], pose[3:6], mtx, dist)
		imagepoints_finetuned[i*2], imagepoints_finetuned[2*i+1] = imgpoints2[0][0][0], imgpoints2[0][0][1]

	return (imagepoints_finetuned)
	
def get_printed(bbox, frame, framenumber, pose):

	pts={0: [45.0, 256.5, 71.0], 1: [-45.0, 256.5, 71.0], 2: [56.0, 186.7, 89.0], 3: [-56.0, 186.7, 89.0], 4: [51.0, 240.0, 149.5], 5: [-51.0, 240.0, 149.5], 6: [0.0, 260.0, 41.5], 7: [-47.5, 43.6, 104.7], 8: [-57.0, 57.8, 145.5], 9: [-54.3, 135.0, 149.8], 10: [-66.4, 220.0, 19.8], 11: [-43.0, 68.0, 165.6], 12: [57.0, 57.8, 145.5], 13: [54.3, 135.0, 149.8], 14: [43.0, 68.0, 165.6], 15: [47.5, 43.6, 104.7], 16: [66.4, 220.0, 19.8], 17: [44.2, 61.5, 152.4], 18: [-44.2, 61.5, 152.2], 19: [49.0, 56.0, 116.5], 20: [-49.0, 56.0, 116.5], 21: [0.0, 0.0, 38.0], 22: [33.0, 36.0, 83.5], 23: [-33.0, 36.0, 83.5]}
	#mtx = np.array([[1324.110551, 0.000000, 993.993108], [0.000000, 1324.110210, 621.997610],[  0. ,   0. ,   1. ]])
	#dist = np.array([[-0.401747, 0.148985, -0.008159, -0.006626, 0.000000]]) 
	mtx = np.array([[1515.104, 0.000000, 665.102], [0.000000, 1523.561, 342.34],[  0. ,   0. ,   1. ]])
	dist = np.array([[-0.27888534, -0.15845886, -0.00238336,  0.00136916,  1.09498805]])
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

	#imagepoints_finetuned = np.empty((len(points)*2), dtype=np.float)

	for i in range(24):
		ponting = np.array([[pts[i]]], dtype=np.float)
		imgpoints2, _ = cv2.projectPoints(ponting, pose[0:3], pose[3:6], mtx, dist)
		print ("well hello",imgpoints2[0][0])
		x,y = int(imgpoints2[0][0][0]+bbox['bbox'][0]), int(imgpoints2[0][0][1]+bbox['bbox'][1])
		print (x,y)
		cv2.circle(frame, (x,y), 5, (0,0,255), -1)

#	cv2.imwrite('/home/shantam/Documents/'+str(framenumber)+'.jpg', frame)

def flip(pos):
	if pos[2]<0:
		pos*=-1
	return pos

def distance(last_pos, current_pos):
	# if last_pos[2]<0:
	# 	last_pos*=-1
	# if current_pos[2]<0:
	# 	current_pos*=-1

	val=0
	for i in range(0,3):
		val += (last_pos[i]-current_pos[i])*(last_pos[i]-current_pos[i])

	val=math.pow(val,0.5)
	return val

def compute_pose(bbox, frame, framenumber, points, imagepoints):
	#pts={0: [49.0, 56.0, 116.5], 1: [-57.0, 57.8, 145.5], 2: [0.0, 0.0, 38.0], 3: [51.0, 240.0, 149.5], 4: [-33.0, 36.0, 83.5], 5: [45.0, 256.5, 71.0], 6: [66.4, 220.0, 19.8], 7: [47.5, 43.6, 104.7], 8: [-54.3, 135.0, 149.8], 9: [-43.0, 68.0, 165.6], 10: [-44.2, 61.5, 152.2], 11: [0.0, 260.0, 41.5], 12: [-47.5, 43.6, 104.7], 13: [-45.0, 256.5, 71.0], 14: [56.0, 186.7, 89.0], 15: [43.0, 68.0, 165.6], 16: [33.0, 36.0, 83.5], 17: [-66.4, 220.0, 19.8], 18: [57.0, 57.8, 145.5], 19: [54.3, 135.0, 149.8], 20: [44.2, 61.5, 152.4], 21: [-49.0, 56.0, 116.5], 22: [-51.0, 240.0, 149.5], 23: [-56.0, 186.7, 89.0], 24: [45.0, 125.0, 117.5]}

	global weights
	pts={0: [45.0, 256.5, 71.0], 1: [-45.0, 256.5, 71.0], 2: [56.0, 186.7, 89.0], 3: [-56.0, 186.7, 89.0], 4: [51.0, 240.0, 149.5], 5: [-51.0, 240.0, 149.5], 6: [0.0, 260.0, 41.5], 7: [-47.5, 43.6, 104.7], 8: [-57.0, 57.8, 145.5], 9: [-54.3, 135.0, 149.8], 10: [-66.4, 220.0, 19.8], 11: [-43.0, 68.0, 165.6], 12: [57.0, 57.8, 145.5], 13: [54.3, 135.0, 149.8], 14: [43.0, 68.0, 165.6], 15: [47.5, 43.6, 104.7], 16: [66.4, 220.0, 19.8], 17: [44.2, 61.5, 152.4], 18: [-44.2, 61.5, 152.2], 19: [49.0, 56.0, 116.5], 20: [-49.0, 56.0, 116.5], 21: [0.0, 0.0, 38.0], 22: [33.0, 36.0, 83.5], 23: [-33.0, 36.0, 83.5]}
	'''order of pts: rear light (left), rear light (right), greenlow corner left, greenlow corner right, top corner rear left, top corner rear right, rear center, indicator light right, 
	right mirror, right center pole top, right wheel, top corner front right, left mirror, left center pole top, top corner front left , indicator light left, left wheel, Wind shield(top left)
	Wind shield(top right), Wind shield(bottom left), Wind shield(bottom right), front bonet, head light left, Head light right'''

	#mtx = np.array([[1324.110551, 0.000000, 993.993108], [0.000000, 1324.110210, 621.997610],[  0. ,   0. ,   1. ]])
	#dist = np.array([[-0.401747, 0.148985, -0.008159, -0.006626, 0.000000]]) 
	mtx = np.array([[1515.104, 0.000000, 665.102], [0.000000, 1523.561, 342.34],[  0. ,   0. ,   1. ]])
	dist = np.array([[-0.27888534, -0.15845886, -0.00238336,  0.00136916,  1.09498805]]) 
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

	worldpoints = np.zeros([len(points),3])
	twist = np.zeros([len(points),2])

	j=0
	for i in points:
		worldpoints[j] = pts[i]
		j+=1

	for i in range(len(imagepoints)):
		twist[i] = [float(imagepoints[i][0]), float(imagepoints[i][1])]

	_, rvecs, tvecs = cv2.solvePnP(worldpoints, twist, mtx, dist)
	tvecs = flip(tvecs)

	mean_error = 0
	for i in range(len(points)):
		imgpoint = np.array([float(imagepoints[i][0]), float(imagepoints[i][1])])
		ponting = np.array([[pts[points[i]]]], dtype=np.float)
		imgpoints2, _ = cv2.projectPoints(ponting, rvecs, tvecs, mtx, dist)
		error = cv2.norm(imgpoint, imgpoints2[0][0], cv2.NORM_L2)/len(imgpoints2)
		mean_error += error
	old_error = mean_error/len(points)

	placeholder = np.array([0, 0, 0, 0, 0, 0], dtype=np.float)
	for i in range(len(rvecs)):
		placeholder[i]=rvecs[i]
		placeholder[i+3]=tvecs[i]

	#get_printed(bbox, frame, framenumber, placeholder)

	twist_reshaped = twist.reshape(-1)
	weights=np.ones((len(twist_reshaped)))
	pose_opt = leastsq(reproject_error, placeholder, args = (points, twist_reshaped), maxfev=1000)
	pose_opt = np.asarray(pose_opt)
	tvecs_opt, rvecs_opt = pose_opt[0][3:6], pose_opt[0][0:3]
	tvecs_opt = flip(tvecs_opt)

	mean_error = 0
	for i in range(len(points)):
		imgpoint = np.array([float(imagepoints[i][0]), float(imagepoints[i][1])])
		ponting = np.array([[pts[points[i]]]], dtype=np.float)
		imgpoints2, _ = cv2.projectPoints(ponting, rvecs_opt, tvecs_opt, mtx, dist)
		error = cv2.norm(imgpoint, imgpoints2[0][0], cv2.NORM_L2)/len(imgpoints2)
		mean_error += error
	new_error = mean_error/len(points)

	placeholder = np.array([0, 0, 0, 0, 0, 0], dtype=np.float)
	for i in range(len(rvecs)):
		placeholder[i]=rvecs_opt[i]
		placeholder[i+3]=tvecs_opt[i]
	#get_printed(bbox, frame, framenumber+1000, placeholder)

	print ("idhar dekho", rvecs, "\n", tvecs, "\n", rvecs_opt, "\n", tvecs_opt)
	print ("translation", old_error - new_error, old_error, new_error)
	#time.sleep(1)

	if new_error<=old_error:
		return tvecs_opt
	else:
		return tvecs

def format_save(out_boxes,out_classes,frame):
	for i in range(len(out_boxes)):
		x,y,w,h = out_boxes[i]
		cv2.rectangle(frame, (int(x-w/2),int(y-h/2)),(int(x+w/2),int(y+h/2)),(255,0,0))
		cv2.putText(frame, str(out_classes[i].decode("utf-8")), (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0))
		out.write(frame)

def draw_counts(frame,counts):

	#counts['up_count']
	#counts['down_count']

	up_count_dict = {k:counts['up_count'][k] for k in sorted(counts['up_count'])}
	down_count_dict = {k:counts['down_count'][k] for k in sorted(counts['down_count'])}
	classes = list(up_count_dict.keys())
	#print("Speed\n",speed)


	up_counts = list(up_count_dict.values())
	down_counts = list(down_count_dict.values())

	cv2.putText(frame, "Dn Count: " + str(up_counts), (10, 40), cv2.FONT_HERSHEY_SIMPLEX,  0.8, (255, 255, 0), 2, cv2.LINE_AA)
	cv2.putText(frame, "Up Count: " + str(down_counts), (10, 80), cv2.FONT_HERSHEY_SIMPLEX,  0.8, (0, 255, 255), 2, cv2.LINE_AA)
	cv2.putText(frame, "Classes: " +str(classes), (10, 150),cv2.FONT_HERSHEY_SIMPLEX,  0.8, (255, 0 , 0), 2, cv2.LINE_AA)

	return frame

def draw_boxes(frame,class_boxes, speed, onlyspeed):

	for item in class_boxes:
		classname = item['classname']
		bbox = item['bbox']
		id_number = item['id']
		#speed = item['speed']

		#Convert to Kilometer per hour and round to next integer
		#speed = speed * 3.6
		#speed = round(speed)

		bbox = list(map(int,bbox))

		x1,y1,x2,y2 = bbox
		#print ("weell", x1, x2, y1, y2)
		#time.sleep(1)

		#cv2.line(frame,count_line[0],count_line[1],(0,0,0),2)
		cropped = frame[y1:y2, x1:x2]
		cv2.rectangle(frame, (x1,y1),(x2,y2),(255, 255, 0),3)
		cv2.putText(frame, "ID:"+str(id_number) , (x2,y2),cv2.FONT_HERSHEY_SIMPLEX,  0.8, (0, 0 , 0),2, cv2.LINE_AA)
		cv2.putText(frame, str(classname.decode("utf-8")), (int(x1), int(y1)), cv2.FONT_HERSHEY_COMPLEX,1.15, (0, 255, 255),2)
		if (onlyspeed==True):
			cv2.putText(frame, "Speed:"+str(round(speed, 2))+ "m/s" , (x2+50,y2), cv2.FONT_HERSHEY_SIMPLEX,  0.8, (0, 0 , 0),2, cv2.LINE_AA)
		#cv2.putText(frame, "CLASS:"+str(classname) , (x2,y2),cv2.FONT_HERSHEY_SIMPLEX,  0.8, (0, 0 , 0), 1, cv2.LINE_AA)
		#cv2.putText(frame, "Speed:"+ str(speed) +' Km/hr' , (x2+10,y2+10),cv2.FONT_HERSHEY_SIMPLEX,  0.8, (0, 0 , 0), 2, cv2.LINE_AA)
	return frame, cropped


#cap = cv2.VideoCapture('/home/shantam/Documents/Programs/PoseEstimation/autospeed/clip 3/clip_3.avi')
cap = cv2.VideoCapture('/media/shantam/D6E430F0E430D509/elcita_video_share_Sept2019/sample_videos_new_cameras/teknic_2/13.avi')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('/media/shantam/D6E430F0E430D509/elcita_video_share_Sept2019/sample_videos_new_cameras/speed_vids/teknic_2/teknic_13.avi',fourcc, 20.0, (1280,720))
#cap.set(cv2.CAP_PROP_FPS, 30)
print("fps rate: ", cv2.CAP_PROP_FPS)
txt = open('/media/shantam/D6E430F0E430D509/elcita_video_share_Sept2019/sample_videos_new_cameras/speed_vids/teknic_2/road_points.csv', "a")
writer = csv.writer(txt, delimiter=',')
#fourcc = cv2.VideoWriter_fourcc(*'DIVX')
#out = cv2.VideoWriter('test_out_4.avi', fourcc, 15.0, (int(cap.get(3)), int(cap.get(4))))
	
#yolo = YOLOV3("cfg/yolo_2k_reanchored.cfg","weights/yolo_2k_reanchored_70000.weights","cfg/2k_aug.data")

count_line = [(650,400),(1000,500)]
sline_1_203 = [(1065,514),(524,366)]
sline_2_203 = [(1038,704),(151,406)]

a = analytics_rbc(count_line,sline_1_203,sline_2_203)
model = hourglass_single.Hourglass()

tracked = []
last = []
tracking = []
img_pix_kps = []
world_pix_ori = []

framecount = 0 
frame_number=0

while True:
	#global tracked, last, tracking
	found = 0

	t = 1.0/20.0 #45.0/900.0#25.0/497.0#10.0/191.0#7.0/125.0

	if frame_number==0:
		ret, frame = cap.read()
		print (frame.shape)
		#time.sleep(10)
		prev_frame = frame.copy()

	else:
		prev_frame = frame
		ret, frame = cap.read()


	if frame is None:
		break
		txt.close()

	if frame is not None:
		#cv2.imwrite('temp/image.jpg',frame)
		#out_scores, out_boxes, out_classes = yolo.detect_image(frame)
		class_boxes = a.run_analytics(frame,300)
		framecount+=1

		for i in class_boxes:
			print ("ndjekbdje3", i['id'], framecount)
			if not(any(i['id'] == x  for x in tracked)):
				frame, cropped = draw_boxes(frame, [i], 0, False)
				#cropped = cv2.imread('/home/shantam/Documents/Programs/hourglasstensorlfow/images/cropped0.jpg')
				ht, wd = cropped.shape[0], cropped.shape[1]
				if (ht>100 and wd>100) or (ht>150) or (wd>150):
					cropped = np.array(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
					#cropped = cv2.resize(cropped, (256,256))
					cropped_new = np.moveaxis(cropped, 2, 0)

					pointers, points = model.forward_pass(cropped_new)
					#print ("printing", pointers, points)

					transformed_points=[]
					for point in points:
						x,y = int(point[0]+i['bbox'][0]), int(point[1]+i['bbox'][1])
						transformed_points.append([x,y])
						cv2.circle(frame, (x,y), 5, (0,0,255), -1)
						#print ("pointssss", x,y)

					if len(points)>=4:
						tracked.append(i['id'])
						print ("wait", len(transformed_points))
						#time.sleep(3)
						position = compute_pose(i, frame, frame_number, pointers, transformed_points)
						#txt.write(str(position)+" \n")
						
						back=0
						front=0
						for idx in pointers:
							if (idx == 0) or (idx == 1) or (idx == 6):
								back+=1
							elif (idx==17) or (idx==18) or (idx==19) or (idx==20) or (idx==21):
								front+=1

						if back>=2:
							tracking.append({'track_id': i['id'], 'detections':0, 'lastpose': position, 'speed':0, 'currentpose': position, 'computes': 0, 'orientation': 'away', 'lastframe':framecount, 'currentframe': framecount, 'latest':0})
							print ("orientation", "away", back, front, position)
						elif front>=3:
							tracking.append({'track_id': i['id'], 'detections':0, 'lastpose': position, 'speed':0, 'currentpose': position, 'computes': 0, 'orientation': 'towards', 'lastframe':framecount, 'currentframe': framecount, 'latest':0})
							print ("orientation", "towards", back, front, position)
						else:
							tracking.append({'track_id': i['id'], 'detections':0, 'lastpose': position, 'speed':0, 'currentpose': position, 'computes': 0, 'orientation': 'unknown', 'lastframe':framecount, 'currentframe': framecount, 'latest':0})
							print ("orientation", "unknown", back, front, position)	

						#cv2.imshow("initial", frame)
						#cv2.waitKey(10)
						
			elif (i['id'] == x for x in tracked):
				frame, cropped = draw_boxes(frame, [i], 0, False)
				ht, wd = cropped.shape[0], cropped.shape[1]

				for j in tracking:

					if j['track_id']==i['id']:
						frame, cray = draw_boxes(frame, [i], j['latest'], True)
						j['detections']+=1

						if j['detections']==2:
							print ("5 hoagye", j["detections"], framecount)
							if (ht>100 and wd>100) or (ht>150) or (wd>150):
								j['computes']+=1
								cropped = np.array(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
								cropped_new = np.moveaxis(cropped, 2, 0)
								pointers, points = model.forward_pass(cropped_new)
								print ("length", len(points))

								transformed_points=[]
								for point in points:
									#print (point)
									#x,y = point[0], point[1]
									#cv2.circle(frame, (x,y), 5, (0,0,255), -1)

									x,y = int(point[0]+i['bbox'][0]), int(point[1]+i['bbox'][1])
									transformed_points.append([x,y])
									cv2.circle(frame, (x,y), 5, (0,0,255), -1)
									#print ("pointssss", x,y)

								# cv2.imshow("initial", frame)
								# cv2.waitKey(10)

								if len(points)>=4:								
									position = compute_pose(i, frame, frame_number, pointers, transformed_points)
									writer.writerow([str(position[0]), str(position[1]), str(position[2])])
									# for every 40th frame, get the points (image and world)and add to list of points which will be used to compute homography
									if (i['id'] == 1) and (frame_number % 150 == 0) and (len(world_pix_ori) <= 8):
										#24 keypoints for first bounding box (1st auto) out of all boxes detected. To get 1st kp for 1st object, x[0][0].
										img_pix_kps.append(transformed_points)
										# world coordinates of origin for the same auto.
										world_pix_ori.append(position)
									if len(world_pix_ori) == 8:
										#get appropriate keypoints as per view, transform origin 3d coords to that keypoint and pick x and y locations to create homography.
										# pointers is the list of indices of points detected, denoting which keypoint was detected.
										print("Keypoint chosen for homography:", pointers[0])
										req_img_coords_kp = [img_pix_kps[i][pointers[0]] for i in range(len(img_pix_kps))]
										# Done: get world coords for the bonet points from world_pix_ori
										for i in range(len(world_pix_ori)):
											req_world_coords_kp[i] = [sum(x) for x in zip(world_pix_ori[i][:2], pts[pointers[0]][:2])]

										array_dump("arrays.txt", req_img_coords_kp, req_world_coords_kp)



										# TODO: compute homography

										# compute_homography(req_img_coords_kp, req_world_coords_kp)


									#writer.writerow(zip([str(position[0]), str(position[1]), str(position[2])], transformed_points))
									#txt.write(str(position)+" \n")
									#print ("POSITIONSSSS", position)
									#time.sleep(2)

									if (j['orientation']=='away') and (position[2]>=j['lastpose'][2]):

										if ((abs(position[2]-j['lastpose'][2]))<(1500+(framecount-j['lastframe'])*20)) and ((abs(position[2]-j['lastpose'][2]))>50): #(framecount-j['lastframe'])*20):
										#if ((abs(position[2]-j['lastpose'][2]))<500):

											j['currentpose'] = position
											j['currentframe'] = framecount
											j['detections'] = 0
											print ("poses here", j['lastpose'], j['currentpose'])

											speed = distance(j['lastpose'], j['currentpose'])/(float(j['currentframe']-j['lastframe'])*t)
											if speed < 2000:
												j['speed'] += speed/100.0
												j['lastpose'] = position
												extra, unnecessary = draw_boxes(frame, [i], j['speed']/j['computes'], True)
												#cv2.imshow("speeded", extra)
												print ("\n", 'speed', j['speed']/j['computes'],  j['track_id'], speed, "going away", j['currentframe'], j['lastframe'])
												j['lastframe'] = framecount
												j['latest'] = j['speed']/j['computes']
												print ("this one", j['latest'], j['computes'])
												#time.sleep(0.5)

											else:
												j['detections']-=1
												j['computes']-=1

										else:
											j['detections']-=1
											j['computes']-=1

									elif (j['orientation']=='towards') and (position[2]<=j['lastpose'][2]):

										#if (abs(position[2]-j['lastpose'][2]))<500:
										print ("difference in z", abs(position[2]-j['lastpose'][2]))
										if ((abs(position[2]-j['lastpose'][2]))<(1500+(framecount-j['lastframe'])*20)) and ((abs(position[2]-j['lastpose'][2]))>50): #(framecount-j['lastframe'])*20):
											j['currentpose'] = position
											j['currentframe'] = framecount
											j['detections'] = 0
											print ("poses here", j['lastpose'], j['currentpose'])

											speed = distance(j['lastpose'], j['currentpose'])/(float(j['currentframe']-j['lastframe'])*t)
											#j['speed'] += speed/100.0
											if speed<2000:
												j['speed'] += speed/100.0
												j['lastpose'] = position
												extra, unnecessary = draw_boxes(frame, [i], j['speed']/j['computes'], True)
												#v2.imshow("speeded", extra)
												print ("\n", 'speed', j['speed']/j['computes'],  j['track_id'], len(points), speed, "coming near", j['currentframe'], j['lastframe'])
												j['lastframe'] = framecount
												j['latest'] = j['speed']/j['computes']
												print ("this one", j['latest'], j['computes'])
												#time.sleep(0.5)

											else:
												j['detections']-=1
												j['computes']-=1

										else:
											j['detections']-=1
											j['computes']-=1


									elif (j['orientation']=='unknown'):

										back=0
										front=0
										for idx in pointers:
											if (idx == 0) or (idx == 1) or (idx == 6):
												back+=1
											elif (idx==17) or (idx==18) or (idx==19) or (idx==20) or (idx==21):
												front+=1
										
										if back>=2:
											j['orientation']='away'
										elif front>=3:
											j['orientation']='towards'

										if ((abs(position[2]-j['lastpose'][2]))<(1500+(framecount-j['lastframe'])*20)) and ((abs(position[2]-j['lastpose'][2]))>50): #(framecount-j['lastframe'])*20):
										#if (abs(position[2]-j['lastpose'][2]))<500:
											j['currentpose'] = position
											j['currentframe'] = framecount
											j['detections'] = 0
											print ("poses here", j['lastpose'], j['currentpose'])

											speed = distance(j['lastpose'], j['currentpose'])/(float(j['currentframe']-j['lastframe'])*t)
											#j['speed'] += speed/100.0
											if speed<2000:
												j['speed'] += speed/100.0
												j['lastpose'] = position
												extra, unnecessary = draw_boxes(frame, [i], j['speed']/j['computes'], True)
												#cv2.imshow("speeded", extra)
												print ("\n", 'speed', j['speed']/j['computes'],  j['track_id'], len(points), speed, "unknown orientation", j['currentframe'], j['lastframe'])
												j['lastframe'] = framecount
												j['latest'] = j['speed']/j['computes']
												print ("this one", j['latest'], j['computes'])
												#time.sleep(0.5)

											else:
												j['detections']-=1
												j['computes']-=1
										else:
											j['detections']-=1
											j['computes']-=1

									else:
										j['detections']-=1
										j['computes']-=1
									
								else:
									j['detections']-=1
									j['computes']-=1
							else:
								j['detections']-=1


		for x in tracked:
			if not(any(x == m['id'] for m in class_boxes)):
				tracked.remove(x)

		print ("so far", tracked)
		#time.sleep(.2)
		#for k in tracked:


		#frame = draw_boxes(frame,class_boxes)
		#frame = draw_counts(frame,counts)
		#im = cv2.resize(frame,(740,580))
		if frame_number!=0:
			# cv2.imshow('counts',frame)
			# cv2.waitKey(10)
			out.write(frame)
		#out.write(frame)
		#format_save(out_boxes,out_classes,frame)
		frame_number+=1
