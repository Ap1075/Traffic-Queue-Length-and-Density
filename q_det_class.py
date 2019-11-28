import cv2
import numpy as np
from operator import itemgetter
import math


"""Get input and dst arrays from auto_single_speed_irls_kyu to compute homography matrix."""

# RBC gate measurements.
inpt_array = [[1421, 895], [1454, 988], [1149, 618], [1116, 592], [772, 785], [
    838, 735]]  # gratings 1 and 2, sedan1 , ace , sedan 2 rear and front
dst_array = [[0.45, 15.8], [0, 17], [4.5, 2.2],
    [7.7, 0], [7.5, 12.4], [7.5, 8.8]]


class q_lenner():

    def __init__(self, mask_img, input_arr, dst_arr, scale=2.5):
        self.old_frame = None
        self.frame_no = 0
        self.old_frame_no = 0
        self.queue_ind = False
        self.stat_greater_count = 0
        self.dyn_greater_count = 0
        self.old_q = []
        self.sd = 0
        self.medd = 0
        # self.video_path = video_path
        self.mask_img = cv2.medianBlur(mask_img, 51)
        # input and dst arrays are for homography estimation. Measured values in img and world coordinates.
        self.input_array = input_arr
        self.dst_array = dst_arr
        self.scale = scale
        self.homat = (cv2.findHomography(np.array(self.input_array),
                      np.array(self.dst_array), cv2.LMEDS, 5))*self.scale

    def q_measure(self, homat, src, dst):
        """Measures distance between extreme features detected."""

        # src and dst must be of type (x,y,1)
        xnew1, ynew1, znew1 = homat.dot(np.array(src))
        xnew2, ynew2, znew2 = homat.dot(np.array(dst))
        distance = math.sqrt(((xnew1-xnew2)**2) +
                             ((ynew1-ynew2)**2) + ((znew1-znew2)**2))

        return distance

    def visualize(self, frame, stat, dyn):
        """Visualizes detected features and queue length measurements. Enabled through run method argument."""

        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (1000, 850)
        # bottomRightCornerOfText = (1000, 870)
        fontScale = 1
        fontColor = (255, 255, 255)
        lineType = 2
        if self.queue_ind:
            cv2.putText(frame, "Queue Length: {0} metres".format(
                self.medd), (1000, 800), font, 1.5, (255, 0, 255), 3)
            if len(self.old_q) > 10 or (max(self.old_q) - min(self.old_q) > 10):
                self.old_q = []

        for corn in stat:
                cv2.circle(frame, (corn[1], corn[0]), 7,
                           (255, 0, 0), -1)         # blue stat
        for corn in dyn:
            cv2.circle(frame, (corn[1], corn[0]), 7, (0, 0, 255), -1)  # red dyn

        cv2.putText(frame, 'Static Corners: {0}, Dynamic Corners: {1}'.format(len(stat), len(dyn)),
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)

    def corns_neighbourhood(self, feats):
        """Creates a neighbourhood of size 3x3 around the detected features(corners)."""

        # contains windows for each corner in the frame (theres a win(dow) for each corner)
        corn_neigh = []
        for i, corner in enumerate(feats):
            x, y = corner.ravel()
            # taking 3x3 neighbourhood around the corner pixel
            corn_neigh.append([y-1, y+1, x-1, x+1])
        return corn_neigh

    def run(self, sframe, visualize=False):
        """Runs the queue length estimation code for each frame."""

        try:
            frame = sframe.get_image()
            self.frame_no += 1  
        except Exception as ex:
            print("Problem while reading input frame.\n")
            print(ex)

        try:
            gr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            greyed = cv2.bitwise_and(frame, frame, mask = self.mask_img)
            greyed = cv2.cvtColor(greyed, cv2.COLOR_BGR2GRAY)
            corns = cv2.goodFeaturesToTrack(greyed, 100, 0.5, 10)
            corns = np.int0(corns)

            # get neighbourhood around corners, required to distinguish between static and dynamic corners.
            corn_neigh= self.corns_neighbourhood(corns)
        except Exception as ex:
            print("Problem detected with input frame.\n")
            print(ex)

        if self.old_frame is not None:
            self.old_frame_no += 1
            # list containing max val for each window around a corner for consecutive frames.
            max_val = []
            stat = []
            dyn = []
            for i,win in enumerate(corn_neigh):
                max_val.append(np.max(np.subtract(gr[win[0]:win[1], win[2]:win[3]], self.old_frame[win[0]:win[1], win[2]:win[3]])))
                thresh = 0.5*(np.max(gr[win[0]:win[1], win[2]:win[3]]) - np.min(gr[win[0]:win[1], win[2]:win[3]]))
                # print("threshold", thresh)
                # if win[0]>260 and win[2]<1450:
                if win[0]>350 and win[2]<1400:
                    stat.append([win[0]+1, win[2]+1]) if max_val[i]<=thresh else dyn.append([win[0]+1, win[2]+1])
            
            if ((len(stat) > 10) or (len(stat)>len(dyn) and len(dyn)<5)):                       ## after false positives are gone, a simpler if.
                self.stat_greater_count +=1
                if self.stat_greater_count > 10:
                    self.queue_ind = True
                    stat = sorted(stat, key=itemgetter(1))
                    if len(stat) == 1:
                        return self.medd
                    # cv2.circle(frame, tuple((stat[-2][1], stat[-2][0])), 15,(0,255,0),-1)
                    # cv2.circle(frame, tuple((stat[0][1], stat[0][0])), 15,(0,255,0),-1)
                    self.medd = round(self.q_measure(self.homat, [stat[0][1], stat[0][0],1], [stat[-2][1], stat[-2][0],1]), 3)
                    if len(self.old_q)>2 and  (self.medd < np.mean(self.old_q)-2*np.std(self.old_q)):
                        self.medd = round(np.mean(self.old_q),2)
    
                    if len(self.old_q)>10 or (max(self.old_q) - min(self.old_q)>10):
                        self.old_q = []
                    else:
                        self.old_q.append(self.medd)
                        # print("queue", medd)
                    self.dyn_greater_count = 0
                    self.stat_greater_count = 0

            elif len(dyn) > 1.5*len(stat) or len(stat)<10:
                self.dyn_greater_count +=1
                # sd =0
                if self.dyn_greater_count > 15:
                    self.queue_ind = False
                    # old_q = []
                    self.stat_greater_count = 0
                    self.dyn_greater_count = 0
                # print(stat_greater_count)
            if (len(self.old_q) > 10 or (max(self.old_q) - min(self.old_q) > 10)):
                self.old_q = []

            if visualize:
                self.visualize(frame, stat, dyn)
            
        self.old_frame = gr

        return self.medd