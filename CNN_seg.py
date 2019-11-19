#from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import cv2
import sys
import numpy as np
from skimage import segmentation
from skimage import color
import torch.nn.init
from PIL import Image

use_cuda = torch.cuda.is_available()
if use_cuda:
    print("Running on GPU.")

parser = argparse.ArgumentParser(description='PyTorch Unsupervised Segmentation')
parser.add_argument('--nChannel', metavar='N', default=100, type=int, 
                    help='number of channels')
parser.add_argument('--maxIter', metavar='T', default=1000, type=int, 
                    help='number of maximum iterations')
parser.add_argument('--minLabels', metavar='minL', default=1, type=int, 
                    help='minimum number of labels')
parser.add_argument('--lr', metavar='LR', default=0.1, type=float, 
                    help='learning rate')
parser.add_argument('--nConv', metavar='M', default=2, type=int, 
                    help='number of convolutional layers')
parser.add_argument('--num_superpixels', metavar='K', default=10000, type=int, 
                    help='number of superpixels')
parser.add_argument('--compactness', metavar='C', default=100, type=float, 
                    help='compactness of superpixels')
parser.add_argument('--visualize', metavar='1 or 0', default=1, type=int, 
                    help='visualization flag')
parser.add_argument('--input', metavar='FILENAME',
                    help='input image file name', required=True)
parser.add_argument('--output', metavar='FILENAME',
                    help='output mask file name', required=True)
# parser.add_argument('--output', metavar='output FILENAME',
                    # help='output image file name', required=True)
args = parser.parse_args()
# cv2.imshow("input_frame",args.input)

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

# CNN model
class MyNet(nn.Module):
    def __init__(self,input_dim):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, args.nChannel, kernel_size=3, stride=1, padding=1 )
        self.bn1 = nn.BatchNorm2d(args.nChannel)
        self.conv2 = []
        self.bn2 = []
        for i in range(args.nConv-1):
            self.conv2.append( nn.Conv2d(args.nChannel, args.nChannel, kernel_size=3, stride=1, padding=1 ) )
            self.bn2.append( nn.BatchNorm2d(args.nChannel) )
        self.conv3 = nn.Conv2d(args.nChannel, args.nChannel, kernel_size=1, stride=1, padding=0 )
        self.bn3 = nn.BatchNorm2d(args.nChannel)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu( x )
        x = self.bn1(x)
        for i in range(args.nConv-1):
            x = self.conv2[i](x)
            x = F.relu( x )
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x

# load image
im = cv2.imread(args.input)
orig_shape = im.shape
# im = cv2.resize(im, (481,321))
im = cv2.resize(im, (570,321)) #aspect ratio
# im = image_resize(im, height=321)
# im = cv2.resize(im, (1280,720))
data = torch.from_numpy( np.array([im.transpose( (2, 0, 1) ).astype('float32')/255.]) )
if use_cuda:
    data = data.cuda()
data = Variable(data)

# slic
# labels = segmentation.slic(im, compactness=args.compactness, n_segments=args.num_superpixels) # slic seg
labels = segmentation.felzenszwalb(im, scale=2, sigma=0.25, min_size=40)
labels = labels.reshape(im.shape[0]*im.shape[1])
u_labels = np.unique(labels)
l_inds = []
for i in range(len(u_labels)):
    l_inds.append( np.where( labels == u_labels[ i ] )[ 0 ] )

# train
# print(data.size(1))
model = MyNet( data.size(1) )
if use_cuda:
    model.cuda()
    for i in range(args.nConv-1):
        model.conv2[i].cuda()
        model.bn2[i].cuda()
model.train()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
label_colours = np.random.randint(255,size=(100,3))
lab4 = np.array([[0,0,0],[255,255,255],[255,0,0],[0,255,0], [0,0,255], [255,255,0],[0,255,0],[0,255,0]])
# label_colours = [[255,255,255],[0,0,0], [0,200,0]]
ii=0
decision = False
for batch_idx in range(args.maxIter):
    # forwarding
    optimizer.zero_grad()
    output = model( data )[ 0 ]
    output = output.permute( 1, 2, 0 ).contiguous().view( -1, args.nChannel )
    ignore, target = torch.max( output, 1 )
    im_target = target.data.cpu().numpy()
    nLabels = len(np.unique(im_target))                                           ##
    # nLabels = 2
    if args.visualize:
        im_target_rgb = np.array([label_colours[ c % len(label_colours) ] for c in im_target]) ##
        im_target_rgb = im_target_rgb.reshape( im.shape ).astype( np.uint8 )
        frr = cv2.cvtColor(im_target_rgb, cv2.COLOR_BGR2HSV)
        if batch_idx >= args.maxIter-10:
            print("number of labels: ", nLabels)
            result = input('Input Label to show and save.\n')
            print("showing label: ",result)
            if str(result) == "done":
                break
            try:
                if int(result) >= nLabels or int(result) == 0:
                    prev_res = result
                    print("Exceeding number of labels, setting to ", nLabels)
                    result = nLabels
                lower1 = np.array(np.unique(frr.reshape(-1, frr.shape[2]), axis=0)[int(result)-1])
                upper1= np.array(np.unique(frr.reshape(-1, frr.shape[2]), axis=0)[int(result)-1])
                mask= cv2.inRange(frr, lower1,upper1)
            except ValueError:
                print("Input numbers only")

            m2 = input('Give 2nd label to add to mask. If not, input 0\n')
            if int(m2) ==0:
                print("No second label added to mask.")
                pass
            else:
                lower2 = np.array(np.unique(frr.reshape(-1, frr.shape[2]), axis=0)[int(m2)-1])
                upper2= np.array(np.unique(frr.reshape(-1, frr.shape[2]), axis=0)[int(m2)-1])
                mask2= cv2.inRange(frr, lower2,upper2)
                mask = cv2.bitwise_or(mask, mask2)

        else:
            lower1 = np.array(np.unique(frr.reshape(-1, frr.shape[2]), axis=0)[ii])
            upper1= np.array(np.unique(frr.reshape(-1, frr.shape[2]), axis=0)[ii])
            mask= cv2.inRange(frr, lower1,upper1)            

        res = cv2.bitwise_and(im_target_rgb, im_target_rgb, mask=mask)

        cv2.imshow( "masked_out", (res) )
        cv2.imshow("mask", mask)
        cv2.imshow( "output", im_target_rgb )

        cv2.waitKey(1)

    # superpixel refinement
    # TODO: use Torch Variable instead of numpy for faster calculation
    for i in range(len(l_inds)):
        labels_per_sp = im_target[ l_inds[ i ] ]
        u_labels_per_sp = np.unique( labels_per_sp )
        hist = np.zeros( len(u_labels_per_sp) )
        for j in range(len(hist)):
            hist[ j ] = len( np.where( labels_per_sp == u_labels_per_sp[ j ] )[ 0 ] )
        im_target[ l_inds[ i ] ] = u_labels_per_sp[ np.argmax( hist ) ]
    target = torch.from_numpy( im_target )
    if use_cuda:
        target = target.cuda()
    target = Variable( target )
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()

    # print (batch_idx, '/', args.maxIter, ':', nLabels, loss.data[0])
    print (batch_idx, '/', args.maxIter, ':', nLabels, loss.item())
    if nLabels <= args.minLabels:##
        print ("nLabels", nLabels, "reached minLabels", args.minLabels, ".")          ##
        break                                                                            ##

# save output image
if not args.visualize:
    output = model( data )[ 0 ]
    output = output.permute( 1, 2, 0 ).contiguous().view( -1, args.nChannel )
    ignore, target = torch.max( output, 1 )
    im_target = target.data.cpu().numpy()
    im_target_rgb = np.array([label_colours[ c % len(label_colours) ] for c in im_target]) ##
    # im_target_rgb = np.array([label_colours[ c % 3 ] for c in im_target])
    im_target_rgb = im_target_rgb.reshape( im.shape ).astype( np.uint8 )
# im_target_rgb = cv2.resize(im_target_rgb, (orig_shape[1],orig_shape[0]))
# im_target_rgb = image_resize(im_target_rgb, height=orig_shape[0]-1, width=orig_shape[1])
im_target_rgb = cv2.resize(im_target_rgb, (1920,1080), interpolation=cv2.INTER_AREA)
im_target_rgb = cv2.medianBlur(im_target_rgb, 7)
cv2.imwrite( "./out/out.png", im_target_rgb )
# cv2.imwrite( args.output, im_target_rgb )
print( "./outs_{0}_{1}_{2}.png".format(args.compactness, args.num_superpixels, args.nConv))
mask = cv2.resize(mask, (1920,1080), interpolation=cv2.INTER_AREA)
mask = cv2.medianBlur(mask, 25)
cv2.imwrite(args.output, mask)
# cv2.imwrite( "./out/new_fb_.png".format(args.compactness, args.num_superpixels, args.nConv), im_target_rgb )