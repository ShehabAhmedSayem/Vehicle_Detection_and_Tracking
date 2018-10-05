import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pickle
from skimage.feature import hog
from sklearn.utils import shuffle

from keras import backend as K
from keras.models import Sequential
from keras.models import Model
from keras.layers import ELU
from keras.layers import Input, Concatenate, concatenate, Convolution2D, MaxPooling2D, UpSampling2D, Lambda
from keras.optimizers import Adam
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2
from keras.models import model_from_json
import simplejson as json
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import random
import math

### Make data frame in Pandas

import pandas as pd

#Reading the first data set from crowd-ai
rootDir = "object-detection-crowdai/"
csvFile = pd.read_csv(rootDir+'labels.csv', header=0)

dataFile = csvFile[(csvFile['Label']!='Pedestrian')].reset_index()
dataFile = dataFile.drop('index', 1)
dataFile = dataFile.drop('Preview URL', 1)
dataFile['Frame'] = './' + rootDir + dataFile['Frame']
dataFile.columns = ['xmin', 'ymin', 'xmax','ymax', 'Frame', 'Label']
dataFile.head(10)
print('first data set len: ',len(dataFile))

#Reading the second data set.
names = ['Frame',  'xmin', 'ymin', 'xmax','ymax', 'occluded', 'Label']
rootDir = "object-dataset/"
csvFile1 = pd.read_csv(rootDir+'labels.csv', delim_whitespace=True, names=names)
dataFile1 = csvFile1[(csvFile1['Label']!=str.lower('Pedestrian'))].reset_index()
dataFile1 = dataFile1.drop('index',1)
dataFile1 = dataFile1.drop('occluded',1)
dataFile1['Frame'] = './' + rootDir + dataFile1['Frame']
dataFile1.tail(10)
print('second dataset len: ',len(dataFile1))


train_samples_per_epoch = 10000
trainBatchSize = 16
imgRow = 630
imgCol = 960

croppedImgRow = 512
croppedImgCol = 512
smooth = 1.

def CropImage(image):
    return cv2.resize(image,(512,512))    

def flip_image(img1,img2):
    coin = np.random.randint(2)    
    if(coin == 0):
        img1 = cv2.flip(img1, 1)
        img2 = cv2.flip(img2, 1)
    return img1,img2
        
def trans_image(image,image1,trans_range=75):
    # Translation augmentation
    
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2

    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    rows,cols,channels = image.shape
   
    image_tr = cv2.warpAffine(image,Trans_M,(cols,rows))
    image_tr1 = cv2.warpAffine(image1,Trans_M,(cols,rows))
    return image_tr, image_tr1


def stretch_image(img,img1,scale_range=50):
    # Stretching augmentation 
    
    tr_x1 = scale_range*np.random.uniform()
    tr_y1 = scale_range*np.random.uniform()
    p1 = (tr_x1,tr_y1)
    tr_x2 = scale_range*np.random.uniform()
    tr_y2 = scale_range*np.random.uniform()
    p2 = (img.shape[1]-tr_x2,tr_y1)

    p3 = (img.shape[1]-tr_x2,img.shape[0]-tr_y2)
    p4 = (tr_x1,img.shape[0]-tr_y2)
    #print(p1,p2,p3,p4)
    pts1 = np.float32([[p1[0],p1[1]],
                   [p2[0],p2[1]],
                   [p3[0],p3[1]],
                   [p4[0],p4[1]]])
    pts2 = np.float32([[0,0],
                   [img.shape[1],0],
                   [img.shape[1],img.shape[0]],
                   [0,img.shape[0]] ]
                   )

    M = cv2.getPerspectiveTransform(pts1,pts2)
    img = cv2.warpPerspective(img,M,(img.shape[1],img.shape[0]))
    img = np.array(img,dtype=np.uint8)    
    
    img1 = cv2.warpPerspective(img1,M,(img1.shape[1],img1.shape[0]))
    img1 = np.array(img1,dtype=np.uint8)
    
    return img,img1

def RandomBrightness(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img)

    scale = random.uniform(.8, 1.25)

    v = np.clip(v * scale, 0, 255, out=v)
    img = cv2.merge((h, s, v))
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    return img

def RotateImage(img,img1,ang_range=40):
    '''
    This function transforms images to generate new images.
    The function takes in following arguments,
    1- Image
    2- ang_range: Range of angles for rotation    
    A Random uniform distribution is used to generate different parameters for transformation

    '''
    # Rotation

    ang_rot = np.random.uniform(ang_range)-ang_range/2
    #print("angle Rot: ",ang_rot)
    coin = np.random.randint(2)
    if coin == 0:
        coin = -1
    rows,cols,ch = img.shape    
    
    Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),coin*ang_rot,1)

    
    img = cv2.warpAffine(img,Rot_M,(cols,rows))    
    
    # Brightness
    img = RandomBrightness(img)    
    
    img1 = cv2.warpAffine(img1,Rot_M,(cols,rows))
    
    return img,img1


def augmentImage(image1,image2):
    
    choice = np.random.randint(4)
    
    image1,image2 = flip_image(image1,image2)
    
    
    if choice == 0:
        return RandomBrightness(image1),image2
    elif choice == 1:
        return stretch_image(image1,image2)
    elif choice == 2:
        return trans_image(image1,image2)
    else:
        return image1, image2
   

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def TrainDataGenerator(dataInfoList,batchSize):
    batch_x, batch_y = [], []
    while True:
        dataInfo = dataInfoList[random.randint(0, 1)]
        row = np.random.randint(len(dataInfo))

        fileName = dataInfo['Frame'][row]
        img = cv2.imread(fileName)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        origShape = img.shape
        img = cv2.resize(img, (imgCol, imgRow))
        
        data = dataInfo[dataInfo['Frame'][row] == dataInfo['Frame']].reset_index()
        data['xmin'] = np.round(data['xmin'] / origShape[1] * imgCol)
        data['xmax'] = np.round(data['xmax'] / origShape[1] * imgCol)
        data['ymin'] = np.round(data['ymin'] / origShape[0] * imgRow)
        data['ymax'] = np.round(data['ymax'] / origShape[0] * imgRow)

        targetImg = np.reshape(np.zeros_like(img[:, :, 2]), (imgRow, imgCol, 1))
        for i in range(len(data)):
            targetImg[int(data.iloc[i]['ymin']):int(data.iloc[i]['ymax']), int(data.iloc[i]['xmin']):int(data.iloc[i]['xmax'])] = 1


        img,targetImg = augmentImage(CropImage(img),CropImage(targetImg)) 

		        
        batch_x.append(img)        
        targetImg = np.reshape(targetImg,(targetImg.shape[0],targetImg.shape[1],1))
        batch_y.append(targetImg)

        if len(batch_x) == batchSize:
            x_array = np.asarray(batch_x)
            y_array = np.asarray(batch_y)
            yield (x_array, y_array)
            batch_x, batch_y = [], []



def CreateModel():
    input_layer = Input((croppedImgRow, croppedImgCol, 3))
    conv0 = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(input_layer)
    conv0 = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(conv0)
    pool0 = MaxPooling2D(pool_size=(2, 2))(conv0)

    conv1 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(pool0)
    conv1 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv4_0 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool4)
    conv4_0 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4_0)

    up5_0 = concatenate([UpSampling2D(size=(2, 2))(conv4_0), conv4], axis=3)
    conv5_0 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up5_0)
    conv5_0 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv5_0)

    up5 = concatenate([UpSampling2D(size=(2, 2))(conv5_0), conv3], axis=3)
    conv5 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up5)
    conv5 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv2], axis=3)
    conv6 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv1], axis=3)
    conv7 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv0], axis=3)
    conv8 = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(conv8)

    conv9 = Convolution2D(1, 1, 1, activation='sigmoid')(conv8)

    model = Model(input=input_layer, output=conv9)

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model


model = CreateModel()

train_samples_per_epoch = (len(dataFile)+len(dataFile1))//trainBatchSize
print(len(dataFile)+len(dataFile1),train_samples_per_epoch)
trainGenerator = TrainDataGenerator((dataFile,dataFile1), trainBatchSize)

weight_save_callback = ModelCheckpoint('./weights/weights.{epoch:02d}-{loss:.4f}.h5', monitor='loss', verbose=2,
                                       save_best_only=False, mode='auto')
model.summary()

model.load_weights('./weights/weights.05--0.9058.h5')

print("Created generator and call backs. Starting training")

model.fit_generator(
    trainGenerator,
    samples_per_epoch=train_samples_per_epoch, nb_epoch=5,
    # validation_data=validGenerator,
    # nb_val_samples=valid_samples_per_epoch,
    callbacks=[weight_save_callback],
    verbose=1
)

model.save_weights('model.h5', True)
with open('model.json', 'w') as file:
    json.dump(model.to_json(), file)