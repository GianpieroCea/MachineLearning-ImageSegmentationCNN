#Neural Networks : MLP

# Before we can feed data to the MLP we need to rescale all the pics
# and then flatten the pictures into 1d vectors
# Code taken from: https://www.kaggle.com/keegil/keras-u-net-starter-lb-0-277
import os
import sys
import random
import warnings
import numpy as np
import pandas as pd
import pathlib

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

# Set some parameters
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
TRAIN_PATH = '/modules/cs342/Assignment2/FullTraining/'
TEST_PATH = '/modules/cs342/Assignment2/FullTesting/'

train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]


# Get and resize train images and masks
X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
print('Getting and resizing train images and masks ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_train[n] = img
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread(path + '/masks/' + mask_file)
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',
                                      preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)
    Y_train[n] = mask

# Get and resize test images
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print('Getting and resizing test images ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img

# Here I flatten all the various different pictures
X_train_pixels= []
for picture in X_train:
    X_train_pixels.append(picture.flatten())
X_train_pixels= np.asarray(X_train_pixels)

Y_train_pixels= []
for picture in Y_train:
    Y_train_pixels.append(picture.flatten())
Y_train_pixels= np.asarray(Y_train_pixels)

X_test_pixels= []
for picture in X_test:
    X_test_pixels.append(picture.flatten())
X_test_pixels= np.asarray(X_test_pixels)

# Train the MLPClassifier from sklearn
# Some of the parameters were done via gridsearch
# modify number of training data to take to tweak
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(20,10,10,20))
mlp.fit(X_train_pixels[:150],Y_train_pixels[:150])



predictions = [mlp.predict(x).reshape((128,128)) for x in X_test_pixels ]

# Resize our predicted test values to their original image sizes
preds_test_resized = []
for i in range(len(predictions)):
    preds_test_resized.append(resize(np.squeeze(predictions[i]), (sizes_test[i][0], sizes_test[i][1]), mode='constant', preserve_range=True))

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

predictions = [rle_encoding(x) for x in preds_test_resized]

# This will be our final dataframe

#import pathlib
import imageio
import numpy as np
import skimage
from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max

df = pd.DataFrame(columns=["ImageID","EncodedPixels"])

# Glob the training data and load a single image path
testing_paths = pathlib.Path('/modules/cs342/Assignment2/FullTesting/').glob('*/images/*.png')
testing = [x for x in testing_paths]
for i,image_path in enumerate(testing):
    image_id = image_path.parts[-3]
    image = imageio.imread(str(image_path))
    series = pd.Series({'ImageID': image_id, 'EncodedPixels': predictions[i]})

    df = df.append(series, ignore_index=True)

print(df)
df.to_csv(path_or_buf='../submission_MLP.csv', index=None)



# #Grid search to find best parameters
# # DO NOT RUN, VERY SLOW!!
# from sklearn.model_selection import GridSearchCV
# parameters={
# 'hidden_layer_sizes':[(20,), (20,10),(20,20,20) ,(30,30,30)],
# 'activation':['logistic', 'tanh', 'relu'],
# 'solver':['lbfgs', 'sgd', 'adam'],
# 'alpha':[0.1,0.01,0.001,0.0001,0.00001],
# 'batch_size':[1,10,50,100,200,500],
# 'learning_rate':['constant', 'invscaling', 'adaptive'],
# 'learning_rate_init':[0.1,0.01,0.001,0.0001,0.00001],
# 'max_iter':[100,200,500,1000]
# }
#
# #Grid Search Cross Validation
# model = GridSearchCV(estimator=mlp,param_grid=parameters,cv=10)
# model.fit(X_train_pixels,Y_train_pixels)
#
# #Displaying best parameters selected via grid search
# print('Best parameter set:')
# best_parameters = model.best_estimator_.get_params()
# print(best_parameters)
#################################################################
# MLP with HoG

# Second way of doing HoG using cv2
# gives better results so to be preferred
# Inspired from:https://www.learnopencv.com/histogram-of-oriented-gradients/
import cv2
from skimage.color import rgb2gray

def hog2(img):

    img = rgb2gray(img)
    img = np.float32(img) / 255.0

    # Calculate gradient
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)

    # Python Calculate gradient magnitude and direction ( in degrees )
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    return mag

def get_hog_pics():
    hog_train = []
    hog_test = []
    for im in X_train:
        hog = hog2(im)
        hog_train.append(hog)
    for im in X_test:
        hog = hog2(im)
        hog_test.append(hog)
    return hog_train,hog_test
hog_train,hog_test = get_hog_pics()

# Here I flatten all the various different pictures
X_train_hog_pixels= []
for picture in hog_train:
    X_train_hog_pixels.append(picture.flatten())
X_train_hog_pixels= np.asarray(X_train_hog_pixels)

Y_train_pixels= []
for picture in Y_train:
    Y_train_pixels.append(picture.flatten())
Y_train_pixels= np.asarray(Y_train_pixels)

X_test_hog_pixels= []
for picture in hog_test:
    X_test_hog_pixels.append(picture.flatten())
X_test_hog_pixels= np.asarray(X_test_hog_pixels)

# Train the MLPClassifier from sklearn
# this time we use the hog picture for training
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(20,40,40,20))
mlp.fit(X_train_hog_pixels[:15],Y_train_pixels[:15])

predictions = [mlp.predict(x).reshape((128,128)) for x in X_test_hog_pixels ]

predictions = [rle_encoding(x) for x in preds_test_resized]

df = pd.DataFrame(columns=["ImageID","EncodedPixels"])

# Glob the training data and load a single image path
testing_paths = pathlib.Path('/modules/cs342/Assignment2/FullTesting/').glob('*/images/*.png')
testing = [x for x in testing_paths]
for i,image_path in enumerate(testing):
    image_id = image_path.parts[-3]
    image = imageio.imread(str(image_path))
    series = pd.Series({'ImageID': image_id, 'EncodedPixels': predictions[i]})

    df = df.append(series, ignore_index=True)

df.to_csv(path_or_buf='../submission_MLP_hog.csv', index=None)

#########################################################################



#MLP with data augmentation

# MLP with Feature Engineering - Data Augmentation
# code inspired by https://www.kaggle.com/shenmbsw/data-augmentation-and-tensorflow-u-net
print('Now running MLP with feature Engineering')

# Data augmentation
# code taken from https://www.kaggle.com/shenmbsw/data-augmentation-and-tensorflow-u-net

import skimage.io
import random
from skimage import transform

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
    
def get_aug_pics():
    aug_train = []
    aug_train_mask = []
    for im,mask in zip(X_train,Y_train):
            aug_pic, aug_mask = data_aug(im,mask[:,:,0])
            aug_train.append(aug_pic)
            aug_train_mask.append(aug_mask)
    return aug_train,aug_train_mask
aug_train,aug_train_mask = get_aug_pics()


# Here I flatten all the various different pictures
X_train_aug_pixels= []
for picture in aug_train:
    X_train_aug_pixels.append(picture.flatten())
for picture in X_train:
    X_train_aug_pixels.append(picture.flatten())
X_train_aug_pixels= np.asarray(X_train_aug_pixels)    

Y_train_aug_pixels= []
for picture in aug_train_mask:
    Y_train_aug_pixels.append(picture.flatten())
for picture in Y_train:
    Y_train_aug_pixels.append(picture.flatten())
Y_train_aug_pixels= np.asarray(Y_train_aug_pixels)

X_test_pixels= []
for picture in X_test:
    X_test_pixels.append(picture.flatten())
X_test_pixels= np.asarray(X_test_pixels)

# Train the MLPClassifier from sklearn
# this time we use the data augmented pictures for training
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(20,40,40,20))
mlp.fit(X_train_aug_pixels[:15],Y_train_aug_pixels[:15])

predictions = [mlp.predict(x).reshape((128,128)) for x in X_test_aug_pixels ]

predictions = [rle_encoding(x) for x in preds_test_resized]

df = pd.DataFrame(columns=["ImageID","EncodedPixels"])

# Glob the training data and load a single image path
testing_paths = pathlib.Path('/modules/cs342/Assignment2/FullTesting/').glob('*/images/*.png')
testing = [x for x in testing_paths]
for i,image_path in enumerate(testing):
    image_id = image_path.parts[-3]
    image = imageio.imread(str(image_path))
    series = pd.Series({'ImageID': image_id, 'EncodedPixels': predictions[i]})

    df = df.append(series, ignore_index=True)


df.to_csv(path_or_buf='../submission_MLP_aug.csv', index=None)
