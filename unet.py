# Uses a modified version of a UNET
# Code inspired by https://www.kaggle.com/keegil/keras-u-net-starter-lb-0-277
# Run on Kaggle netbooks or update Keras = tensorflow to latest version
import os
import sys
import random
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import tensorflow as tf
from keras.layers import merge
from keras.models import Model
from keras.models import Sequential
from keras.layers import BatchNormalization, Deconvolution2D, Conv1D, Conv2D, Conv2DTranspose, UpSampling2D, Lambda, Dense, Activation, Input, Dropout, MaxPooling2D, concatenate

from subprocess import check_output
print(check_output(["ls", "../input/"]).decode("utf8"))


#Funcs
# Define IoU metric, taken from
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

#DICE FUNCTION
from keras import backend as K
smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

# Set some parameters
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
TRAIN_PATH = '../input/stage1_train/'
AUG_PATH ='../input/stage1_train/augs'
TEST_PATH = '../input/stage1_test/'


# #data augmentation
# def data_aug(image,label,angel=30,resize_rate=0.9):
#     flip = random.randint(0, 1)
#     size = image.shape[0]
#     rsize = random.randint(np.floor(resize_rate*size),size)
#     w_s = random.randint(0,size - rsize)
#     h_s = random.randint(0,size - rsize)
#     sh = random.random()/2-0.25
#     rotate_angel = random.random()/180*np.pi*angel
#     # Create Afine transform
#     afine_tf = transform.AffineTransform(shear=sh,rotation=rotate_angel)
#     # Apply transform to image data
#     image = transform.warp(image, inverse_map=afine_tf,mode='edge')
#     label = transform.warp(label, inverse_map=afine_tf,mode='edge')
#     # Randomly corpping image frame
#     image = image[w_s:w_s+size,h_s:h_s+size,:]
#     label = label[w_s:w_s+size,h_s:h_s+size]
#     # Ramdomly flip frame
#     if flip:
#         image = image[:,::-1,:]
#         label = label[:,::-1]
#     return image, label


# def make_data_augmentation(image_ids,split_num):
#     for ax_index, image_id in tqdm(enumerate(image_ids),total=len(image_ids)):
#         image,labels = read_image_labels(image_id)
#         if not os.path.exists("../input/stage1_train/{}/augs/".format(image_id)):
#             os.makedirs("../input/stage1_train/{}/augs/".format(image_id))
#         if not os.path.exists("../input/stage1_train/{}/augs_masks/".format(image_id)):
#             os.makedirs("../input/stage1_train/{}/augs_masks/".format(image_id))

#         # also save the original image in augmented file
#         plt.imsave(fname="../input/stage1_train/{}/augs/{}.png".format(image_id,image_id), arr = image)
#         plt.imsave(fname="../input/stage1_train/{}/augs_masks/{}.png".format(image_id,image_id),arr = labels)

#         for i in range(split_num):
#             new_image, new_labels = data_aug(image,labels,angel=5,resize_rate=0.9)
#             aug_img_dir = "../input/stage1_train/{}/augs/{}_{}.png".format(image_id,image_id,i)
#             aug_mask_dir = "../input/stage1_train/{}/augs_masks/{}_{}.png".format(image_id,image_id,i)
#             plt.imsave(fname=aug_img_dir, arr = new_image)
#            plt.imsave(fname=aug_mask_dir,arr = new_labels)

# def clean_data_augmentation(image_ids):
#     for ax_index, image_id in tqdm(enumerate(image_ids),total=len(image_ids)):
#         if os.path.exists("../input/stage1_train/{}/augs/".format(image_id)):
#             shutil.rmtree("../input/stage1_train/{}/augs/".format(image_id))
#         if os.path.exists("../input/stage1_train/{}/augs_masks/".format(image_id)):
#             shutil.rmtree("../input/stage1_train/{}/augs_masks/".format(image_id))


# image_ids = check_output(["ls", "../input/stage1_train/"]).decode("utf8").split()
# split_num = 10
# make_data_augmentation(image_ids,split_num)
# clean_data_augmentation(image_ids)

# Build U-Net model
inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = Lambda(lambda x: x / 255) (inputs)

c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
c1 = Dropout(0.1) (c1)
c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
c2 = Dropout(0.1) (c2)
c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
c3 = Dropout(0.2) (c3)
c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
c4 = Dropout(0.2) (c4)
c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
c5 = Dropout(0.3) (c5)
c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
c6 = Dropout(0.2) (c6)
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
c7 = Dropout(0.2) (c7)
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
c8 = Dropout(0.1) (c8)
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
c9 = Dropout(0.1) (c9)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss=dice_coef_loss, metrics=[mean_iou])
model.summary()







# Fit model
#earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('model-dsbowl2018-1.h5', verbose=1, save_best_only=True)
results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=10, epochs=10,
                    callbacks=[ checkpointer])






import pandas as pd
from skimage.filters import threshold_otsu
from scipy import ndimage
import pathlib
import imageio
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from scipy import ndimage

model = load_model('model-dsbowl2018-1.h5', custom_objects={'mean_iou': mean_iou, 'dice_coef_loss': dice_coef_loss})

def rle_encoding(x):
    '''
   x: numpy array of shape (height, width), 1 - mask, 0 - background
   Returns run length as list
   '''
    dots = np.where(x.T.flatten()==1)[0] # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return " ".join([str(i) for i in run_lengths])

def RLE_mask(mask,image_id,df):
    '''
   Returns pandas dataframe of each RLE string for each (predicted) nuclei of an image
   '''
    # Deriving individual mask for each object
    labels, numlabels = ndimage.label(mask)     #labels each component with different number
    label_arrays = []
    im_df = pd.DataFrame(columns=["ImageID","EncodedPixels"])
    for n in range(1, numlabels+1, 1):
        label_mask = np.where(labels == n, 1, 0)
        if sum(sum(label_mask))>=10:        # checks if the nuclei is big enough to be considered
            label_arrays.append(label_mask)
        else:
            mask = np.where(labels == n, 0, mask)
            numlabels += -1
    labels, emptyarg = ndimage.label(mask)

    # Watershed by distance to edge
    #distance = ndi.distance_transform_edt(label)
    #local_maxi = peak_local_max(distance, indices=False, labels=label)
    #markers = ndi.label(local_maxi)[0]
    #label = watershed(-distance, markers, mask=label)

    #Adding to df
    im_df = pd.DataFrame(columns=["ImageID","EncodedPixels"])
    for n in range(1, numlabels+1, 1):
            label_mask = np.where(labels == n, 1, 0)
            rle_string = rle_encoding(label_mask)
            series = pd.Series({'ImageID': image_id, 'EncodedPixels': rle_string})
            im_df = im_df.append(series, ignore_index=True)
    return im_df


# MAIN

import pathlib
import imageio
import numpy as np
import matplotlib.pyplot as plt
import skimage
from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max

#predicted test values
preds_test = model.predict(X_test, verbose=1)

# Create list of upsampled test masks
preds_test_upsampled = []
for i in range(len(preds_test)):
    preds_test_upsampled.append(resize(np.squeeze(preds_test[i]), (sizes_test[i][0], sizes_test[i][1]), mode='constant', preserve_range=True))

# This will be our final dataframe
df = pd.DataFrame(columns=["ImageID","EncodedPixels"])

j=0
# RLE encode the test data
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):

    image_id = id_
    image_label = preds_test_upsampled[j]

    #CONVERT PROBABILISTIC PREDICTION TO BINARY
    from skimage.filters import threshold_otsu
    thresh_val = threshold_otsu(image_label)
    newmask = np.where(image_label > thresh_val, 1, 0)

    add_to_df = RLE_mask(newmask,image_id,df)
    df = df.append(add_to_df, ignore_index=True)
    j+=1

print(df)
df.to_csv('CNNsubmission.csv', index=False)



#See stuff
image_label = preds_test_upsampled[0]

#CONVERT PROBABILISTIC PREDICTION TO BINARY
from skimage.filters import threshold_otsu
thresh_val = threshold_otsu(image_label)
print(thresh_val)
mask = np.where(image_label > thresh_val, 1, 0)
mask = image_label
flat = mask.flatten()
newmask=flat.reshape((mask.shape[0],mask.shape[1]))
plt.imshow(newmask)
