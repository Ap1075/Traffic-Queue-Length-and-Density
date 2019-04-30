import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
import math
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from skimage.color import label2rgb 

parser = argparse.ArgumentParser()
parser.add_argument("-p", metavar="Path to image", dest="img_path",required=True,
	                help="String containing image's path", type=str)
kargs = parser.parse_args()

def resizer(*args):
	for i in args:
		print(i)
		cv2.imshow("resized", cv2.resize(i, (740,580)))
	return args

def deginrad(degree):
    return 2*np.pi/360 * degree

def gabor_bank(del_theta, sig_max, frame):		# bank of filters with different theta, sigma and lambda values for orientation, scale and sensitivity to low/high freqs
	filter_bank = []
	wav_min = 4/math.sqrt(2)
	wav_max = math.hypot(frame.shape[0], frame.shape[1])
	n = math.floor(math.log(wav_max/wav_min,2))
	# print("n:", n)
	for wavl in np.arange(0,n-2, 0.5):
		wavlength = (2**wavl)*wav_min
		for theta in range(0, 180-del_theta, del_theta):
			# for sigma in range(0, sig_max, 5):
			kern = cv2.getGaborKernel((17,17), 10, deginrad(theta), wavlength, 0.75, 5, ktype=cv2.CV_32F)
			filter_bank.append(kern)
	return filter_bank

def gabor_apply(frame):
	gabored = cv2.getGaborKernel((17,17), 10.0, np.pi/6, 6.7, 0.75, 5, ktype=cv2.CV_32F)
	# to_be_ret = cv2.filter2D(frame, cv2.CV_8UC3, kernel)
	return cv2.filter2D(frame, cv2.CV_8UC3, gabored)

def get_features(filter_bank, frame):
	features = np.zeros((len(filter_bank), 2), dtype=np.double)
	for k, kernel in enumerate(filter_bank):
		filtered = cv2.filter2D(frame, cv2.CV_8UC3, kernel)
		features[k,0] = filtered.mean()
		features[k,1] = filtered.var()
		features = preprocessing.normalize(features)
		x = features
		# print(features)
		pca = PCA(n_components =2)
		y = pca.fit_transform(features)
	# print(features)
	return features

def filters_as_features(filter_bank, frame):
	print("number of filters:",len(filter_bank))
	features = np.zeros((1080, 1920, 3, len(filter_bank)), dtype=np.double)
	for k, kernel in enumerate(filter_bank):
		filtered = cv2.filter2D(frame, cv2.CV_8UC3, kernel)
		filtered = cv2.GaussianBlur(filtered, (17,17), 0)
		# filtered = cv2.medianBlur(filtered, 25)
		features[:,:,:,k] = filtered
		# print("Filter number added to features", count)
		# features[k,1] = filtered.var()
	features = np.reshape(features, (frame.shape[0]*frame.shape[1]*frame.shape[2], -1))
	features = preprocessing.normalize(features)
	pca = PCA(n_components =1)
	# print(features[0])
	y = pca.fit_transform(features)
	# print(y.shape)
	print((features).shape)
	print(y.shape)
	features = np.reshape( y, (frame.shape[0], frame.shape[1], frame.shape[2]))									############# check the shapes and change.
	return features, y

if __name__ == "__main__":
	frame = cv2.imread(kargs.img_path)
	frame_array = []
	# fpbg=cv2.bgsegm.createBackgroundSubtractorMOG()
	# true_frame = frame
	gr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	# frame = cv2.GaussianBlur(frame, (7,7),0)  
	# closekernel = np.ones((51,51),np.uint8)

	############################################################################
	fb = gabor_bank(30, 11, frame)
	feats, for_km = filters_as_features(fb, frame)
	feats2d = np.reshape(feats, (feats.shape[0]*feats.shape[1], feats.shape[2]))
	print("feats2dshape", feats2d.shape)
	kmeans_cluster = KMeans(n_clusters=2, random_state=0).fit(feats2d)
	cluster_centers = kmeans_cluster.cluster_centers_
	cluster_labels = kmeans_cluster.labels_
	l2r = label2rgb(np.reshape(cluster_labels, (feats.shape[0], feats.shape[1])), colors=["white", "black", "red"])
	cv2.imwrite("./l2r.jpg", l2r)
	print(l2r.shape)
	# hsv = cv2.cvtColor(l2r.astype(int), cv2.COLOR_BGR2HSV)

	# imagem = cv2.bitwise_not(cluster_centers[cluster_labels].reshape(feats.shape[0], feats.shape[1], feats.shape[2]))
	xx = cv2.imread("./l2r.jpg",0)
	(tr, xx) = cv2.threshold(xx, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	print("otsu thresh", tr)
	xx = cv2.bitwise_not(xx)
	print("xx shape:", xx.shape)
	cv2.imshow("saved l2r", xx)
	# cv2.imshow("feats", feats)
	# cv2.imshow("l2r", l2r)
	# plt.show()
	###########################################################################

	filtered_img = gabor_apply(frame)
	print(filtered_img.shape)
	gr_fil = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2GRAY)
	thresh2 = cv2.adaptiveThreshold(gr_fil,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,2)
	cv2.imshow("t2", thresh2)
	med = cv2.medianBlur(thresh2, 25)
	xx = cv2.medianBlur(xx, 51)
	# blur = cv2.GaussianBlur(thresh2, (31,31), 0)														### blur
	thresh3 = cv2.bitwise_not(med)
	# x = thresh3
	kernel = np.ones((3,3),np.float32)/9
	thresh3 = cv2.filter2D(thresh3,-1,kernel)
	thresh3 = cv2.medianBlur(thresh3, 101)
	# thresh3= cv2.morphologyEx(thresh3, cv2.MORPH_CLOSE, closekernel)
	# xx = cv2.bitwise_not(xx)
	newer_fr = cv2.bitwise_and(frame, frame, mask = xx)
	new_fr = cv2.bitwise_and(frame, frame, mask = thresh3)
	cv2.imwrite("./maskforgate.jpg", thresh3)
	cv2.imwrite("./newer_fr.jpg", newer_fr)
	# xx = cv2.bitwise_and(true_frame, true_frame, mask = x)
	# img_float32 = np.float32(frame)
	# dft = cv2.dft(img_float32, flags = cv2.DFT_COMPLEX_OUTPUT)
	# dft_shift = np.fft.fftshift(dft)
	# magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
	inv =  cv2.bitwise_not(filtered_img)
	# fgmask = fpbg.apply(gr)
	corns = cv2.goodFeaturesToTrack(gr, 100, 0.01, 7)
	corns = np.int0(corns)
	corn_neigh=[]						# contains windows for each corner in the frame (theres a win(dow) for each corner)
	for i,corner in enumerate(corns):
		x,y = corner.ravel()
		# cv2.circle(frame, (x,y), 10,255,-1)	
		corn_neigh.append([y-1,y+2, x-1,x+2])  # taking 7x7 neighbourhood around the corner pixel

	frame_array.append(frame)
	old_frame = gr
	lis = [frame, filtered_img, med, new_fr]
	# frame, filtered_img, med, new_fr = resizer(*lis)
	# cv2.imshow("fgmask",fgmask)
	cv2.imshow("frame",frame)
	# cv2.imshow("inv_img",inv)
	# cv2.imshow("filtered_img",filtered_img)
	# cv2.imshow("filtered_img",filtered_img)
	# cv2.imshow("blurred median",med)
	cv2.imshow("applied 1 gabor filter",new_fr)
	cv2.imshow("applied gabor filter bank",newer_fr)
	# cv2.imwrite('./gabor_outs/gabor1.jpg', new_fr)
	# cv2.imwrite('./gabor_outs/gaborbank.jpg', newer_fr)
	# cv2.imshow("applied blurred med",xx)
	# cv2.imshow("fft",magnitude_spectrum)
	cv2.waitKey(0)
	cv2.destroyAllWindows()