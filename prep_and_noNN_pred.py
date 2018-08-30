# First part


#Importing the data
#Adapted from https://www.kaggle.com/stkbailey/teaching-notebook-for-total-imaging-newbies



import pathlib
import imageio
import numpy as np
import matplotlib.pyplot as plt

# Glob the training data and load a single image path
training_paths = pathlib.Path('/modules/cs342/Assignment2/FullTraining').glob('*/images/*.png')
training_sorted = sorted([x for x in training_paths])

masks_paths = pathlib.Path('/modules/cs342/Assignment2/FullTraining').glob('*/masks/*.png')
masks_sorted = sorted([x for x in masks_paths])

testing_paths = pathlib.Path('/modules/cs342/Assignment2/FullTesting').glob('*/images/*.png')
testing_sorted = sorted([x for x in testing_paths])


#Preprocessing and Feature Engineering

#gray version
from skimage.color import rgb2gray
def gray_scale(im):
    im_gray = rgb2gray(im)
    return im_gray
#Here we try to apply HoG tecnique:
from skimage.feature import hog
from skimage import data, exposure

#should apply this to gray scale image

def hog(im):
    fd,hog_image = hog(im, orientations=16, pixels_per_cell=(2, 2),cells_per_block=(3, 3),visualise=True)
    return fd,hog_image

# Second way of doing HoG using cv2
# gives better results so to be preferred
# Inspired from:https://www.learnopencv.com/histogram-of-oriented-gradients/
import cv2

def hog2(img):

    img = rgb2gray(img)
    img = np.float32(img) / 255.0

    # Calculate gradient
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)

    # Python Calculate gradient magnitude and direction ( in degrees )
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    return mag

#Here we try to use thresholding to separate nuclei from background
#taken from https://www.kaggle.com/stkbailey/teaching-notebook-for-total-imaging-newbies
from skimage.filters import threshold_otsu


def otsu(im):
    thresh_val = threshold_otsu(im)
    mask = np.where(im > thresh_val, 1, 0)

    # Make sure the larger portion of the mask is considered background
    if np.sum(mask==0) < np.sum(mask==1):
        mask = np.where(mask, 0, 1)
    return mask

# Data augmentation
# code taken from https://www.kaggle.com/shenmbsw/data-augmentation-and-tensorflow-u-net

import skimage.io
import random
from skimage import transform
def read_image_labels(image_id):
    # most of the content in this function is taken from 'Example Metric Implementation' kernel
    # by 'William Cukierski'
    image_file = "/modules/cs342/Assignment2/FullTraining/{}/images/{}.png".format(image_id,image_id)
    mask_file = "/modules/cs342/Assignment2/FullTraining/{}/masks/*.png".format(image_id)
    image = skimage.io.imread(image_file)
    masks = skimage.io.imread_collection(mask_file).concatenate()
    height, width, _ = image.shape
    num_masks = masks.shape[0]
    labels = np.zeros((height, width), np.uint16)
    for index in range(0, num_masks):
        labels[masks[index] > 0] = index + 1
    return image, labels

def data_aug(image,label,angel=30,resize_rate=0.9):
    flip = random.randint(0, 1)
    size = image.shape[0]
    rsize = random.randint(np.floor(resize_rate*size),size)
    w_s = random.randint(0,size - rsize)
    h_s = random.randint(0,size - rsize)
    sh = random.random()/2-0.25
    rotate_angel = random.random()/180*np.pi*angel
    # Create Afine transform
    afine_tf = transform.AffineTransform(shear=sh,rotation=rotate_angel)
    # Apply transform to image data
    image = transform.warp(image, inverse_map=afine_tf,mode='edge')
    label = transform.warp(label, inverse_map=afine_tf,mode='edge')
    # Randomly corpping image frame
    image = image[w_s:w_s+size,h_s:h_s+size,:]
    label = label[w_s:w_s+size,h_s:h_s+size]
    # Ramdomly flip frame
    if flip:
        image = image[:,::-1,:]
        label = label[:,::-1]
    return image, label

# Idea: maybe image erosion will detach ovderlapping masks
# also tries opening = erosion -> dilation
from skimage.morphology import binary_erosion
from skimage.morphology import erosion
from skimage.morphology import opening

def erode(mask):
    er = binary_erosion(mask, selem=None, out=None)
    return er

def op_morph(mask):
    op = opening(mask)
    return op

# Deriving individual mask for each object

def find_labels(mask):
    labels, numlabels = ndimage.label(mask)
    label_arrays = []
    im_df = pd.DataFrame(columns=["ImageID","EncodedPixels"])
    for n in range(1, numlabels+1, 1):
        label_mask = np.where(labels == n, 1, 0)
        #only takes masks abovea certan threshold
        if sum(sum(label_mask))>=13:
            label_arrays.append(label_mask)
        else:
            mask = np.where(labels == n, 0, mask)
            numlabels += -1
    labels, nlabels = ndimage.label(mask)
    return (labels,numlabels)

#rle encoding
def rle_encoding(x):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = np.where(x.T.flatten()==1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return " ".join([str(i) for i in run_lengths])

def create_row(labels, numlabels, image_id):
    row = pd.DataFrame(columns=["ImageID","EncodedPixels"])
    for n in range(1, numlabels+1, 1):
            label_mask = np.where(labels == n, 1, 0)
            rle_string = rle_encoding(label_mask)
            series = pd.Series({'ImageID': image_id, 'EncodedPixels': rle_string})
            row = row.append(series, ignore_index=True)

    return row

# Prediction without Neural Network
import pandas as pd
import skimage
from scipy import ndimage
from skimage.morphology import watershed
from skimage.feature import peak_local_max
# prediction using no NN
def predict_no_NN():

    #df to output
    df = pd.DataFrame(columns=["ImageID","EncodedPixels"])
    for image_path in testing_sorted:
        im_id = image_path.parts[-3]
        im = imageio.imread(str(image_path))
        im_gray = gray_scale(im)
        mask = otsu(im_gray)
        #this is an additional step,in case remove
        op = op_morph(mask)
        labels,numlabels = find_labels(op)
        row = create_row(labels,numlabels,im_id)
        df = df.append(row, ignore_index=True)

        print('New image added')
    return df
df = predict_no_NN()
df.to_csv(path_or_buf='../submission_no_NN_er.csv', index=None)
