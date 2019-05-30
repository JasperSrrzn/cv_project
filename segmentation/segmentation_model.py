import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import os
import sys
from keras.models import Model
from keras.regularizers import l2
from keras.layers import *
from keras.engine import Layer
from keras.applications.vgg16 import *
from keras.models import *
import keras.backend as K
import tensorflow as tf
from keras.optimizers import *
from utils.get_weights_path import *
from utils.basics import *
from utils.resnet_helpers import *
from utils.BilinearUpSampling import *
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

class segmentation_model(object):

    def __init__(self,pretrained_weights=None):
        input_size = (224,224,3)
        weight_decay=0.
        classes=2
        inputs = Input(input_size)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block1_conv1',
                   kernel_regularizer=l2(weight_decay))(inputs)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block1_conv2',
                   kernel_regularizer=l2(weight_decay))(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)


        x = Conv2D(1024, (3, 3), activation='relu', padding='same', name='block2_conv1',
                   kernel_regularizer=l2(weight_decay))(x)
        x = Conv2D(1024, (3, 3), activation='relu', padding='same', name='block2_conv2',
                   kernel_regularizer=l2(weight_decay))(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)


        x = Conv2D(2056, (3, 3), activation='relu', padding='same', name='block3_conv1',
                   kernel_regularizer=l2(weight_decay))(x)
        x = Conv2D(2056, (3, 3), activation='relu', padding='same', name='block3_conv2',
                   kernel_regularizer=l2(weight_decay))(x)
        x = Conv2D(2056, (3, 3), activation='relu', padding='same', name='block3_conv3',
                   kernel_regularizer=l2(weight_decay))(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)


        x = Conv2D(2056, (3, 3), activation='relu', padding='same', name='block4_conv1',
                   kernel_regularizer=l2(weight_decay))(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2',
                   kernel_regularizer=l2(weight_decay))(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3',
                   kernel_regularizer=l2(weight_decay))(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)


        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1',
                   kernel_regularizer=l2(weight_decay))(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2',
                   kernel_regularizer=l2(weight_decay))(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3',
                   kernel_regularizer=l2(weight_decay))(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)


        x = Conv2D(4096, (7, 7), activation='relu', padding='same', name='fc1', kernel_regularizer=l2(weight_decay))(x)
        x = Dropout(0.5)(x)
        x = Conv2D(4096, (1, 1), activation='relu', padding='same', name='fc2', kernel_regularizer=l2(weight_decay))(x)
        x = Dropout(0.5)(x)

        x = Conv2D(classes, (1, 1), kernel_initializer='he_normal', activation='linear', padding='valid',
                   strides=(1, 1), kernel_regularizer=l2(weight_decay))(x)

        x = BilinearUpSampling2D(size=(32, 32))(x)

        self.model = Model(inputs, x)
        self.model.compile(optimizer=Adam(lr=1e-3), loss=dice_coef_loss)
        if (pretrained_weights):
            self.model.load_weights(pretrained_weights)

    def fit(self,X_train,y_train,X_validation,y_validation,name,epochs=50):
        model_dir = os.path.dirname(os.getcwd()) + '/segmentation/saved_models/'
        callbacks = [ModelCheckpoint(model_dir + name, monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=False),
                     EarlyStopping(patience=20, verbose=1),
                     ReduceLROnPlateau(patience=10, verbose=1),
                     TensorBoard(log_dir='./logs/' + name[:-3], histogram_freq=0, batch_size=32, write_graph=True,
                                 write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                                 embeddings_metadata=None, embeddings_data=None, update_freq='epoch')]
        self.model.fit(x=X_train, y=y_train, epochs=epochs, validation_data=[X_validation, y_validation],
                       callbacks=callbacks)

    def fit_generator(self,X_train,y_train,X_validation,y_validation,datagen,name,epochs=50):
        model_dir = os.path.dirname(os.getcwd()) + '/segmentation/saved_models/'
        callbacks = [ModelCheckpoint(model_dir + name, monitor='val_loss', verbose=1, save_best_only=True,
                                     save_weights_only=False),
                     EarlyStopping(patience=20, verbose=1),
                     ReduceLROnPlateau(patience=10, verbose=1),
                     TensorBoard(log_dir='./logs/' + name[:-3], histogram_freq=0, batch_size=32, write_graph=True,
                                 write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                                 embeddings_metadata=None, embeddings_data=None, update_freq='epoch')]
        datagen.fit(X_train)
        self.model.fit_generator(datagen.flow(x=X_train, y=y_train, batch_size=16), steps_per_epoch=20, epochs=epochs,validation_data=[X_validation, y_validation],
                       callbacks=callbacks)

    def predict(self,X):
        return self.model.predict(X)


    def load_weights(self,path=None):
        self.model.load_weights(path)
