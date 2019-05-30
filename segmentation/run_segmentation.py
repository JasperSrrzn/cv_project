from fcn import fcn
from segmentation_model import segmentation_model
import os
import numpy as np
import matplotlib.pyplot as plt
from unet import Unet
from keras.preprocessing.image import ImageDataGenerator

n_classes = 2
num_epochs = 1
latent_dimension = 10
loss_of_autoencoder = 'xent'
data_dir = os.path.dirname(os.getcwd())+'/output_data/'
model_dir = os.path.dirname(os.getcwd())+'/segmentation/saved_models/'

#name
name = 'unet.h5'
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

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    vertical_flip=True)

model = segmentation_model()
model.fit(X_train=X_train,y_train=Y_train,X_validation=X_val,y_validation=Y_val,name=name,epochs=num_epochs)

best_model = segmentation_model()
best_model.load_weights(model_dir+name)

predictions_test = best_model.predict(X_test)
predictions_train = best_model.predict(X_train)

image1 = X_train[0]
pred = predictions_train[0,:,:,0]
pred = np.reshape(pred,(pred.shape[0],pred.shape[1],1))
segm1 = np.concatenate((pred,pred,pred),axis=-1)
print(segm1.shape)
plt.imshow(image1)
plt.savefig('/content/gdrive/My Drive/first_train_image.png')
plt.imshow(segm1)
plt.savefig('/content/gdrive/My Drive/first_train_segm.png')

image1 = X_test[0]
pred = predictions_test[0,:,:,0]
pred = np.reshape(pred,(pred.shape[0],pred.shape[1],1))
segm1 = np.concatenate((pred,pred,pred),axis=-1)
plt.imshow(image1)
plt.savefig('/content/gdrive/My Drive/first_test_image.png')
plt.imshow(segm1)
plt.savefig('/content/gdrive/My Drive/first_test_segm.png')
