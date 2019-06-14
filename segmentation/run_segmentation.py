from fcn import fcn
from segmentation_model import segmentation_model
from unet import Unet
import os
import numpy as np
import matplotlib.pyplot as plt
from unet import Unet
from keras.preprocessing.image import ImageDataGenerator

n_classes = 2
num_epochs = 1000
data_dir = os.path.dirname(os.getcwd())+'/output_data/'
model_dir = '/content/gdrive/My Drive/segmentation/saved_models/'

#name
name = 'unet_dice_new.h5'

# read train data
X_train = np.load(data_dir+'x_train_img.npy')
Y_train = np.load(data_dir+'y_train_seg.npy')
Y_train = np.reshape(Y_train,(Y_train.shape[0],Y_train.shape[1],Y_train.shape[2],1))

#read val data
X_val = np.load(data_dir+'x_val_img.npy')
Y_val = np.load(data_dir+'y_val_seg.npy')
Y_val = np.reshape(Y_val,(Y_val.shape[0],Y_val.shape[1],Y_val.shape[2],1))

#read test data
X_test = np.load(data_dir+'x_test_img.npy')
Y_test = np.load(data_dir+'y_test_seg.npy')
Y_test = np.reshape(Y_test,(Y_test.shape[0],Y_test.shape[1],Y_test.shape[2],1))

model = Unet(n_filters=16)
model.fit(X_train=X_train,y_train=Y_train,X_validation=X_val,y_validation=Y_val,name=name,epochs=num_epochs)

best_model = Unet(n_filters=16)
best_model.load_weights(model_dir+name)

predictions_test = best_model.predict(X_test)
predictions_train = best_model.predict(X_train)

np.save('/content/gdrive/My Drive/segmentation/test_segmentations_dice_new.npy',predictions_test)
