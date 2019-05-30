import os
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras import backend as K

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

class Unet(object):
    def __init__(self,n_filters,pretrained_weights=None):
        if pretrained_weights==None:
            input_size = (224, 224, 3)
            inputs = Input(input_size)

            conv1 = Conv2D(filters=n_filters, kernel_size=(kernel_size,kernel_size),kernel_initializer='he_normal',padding='same')(inputs)
            conv1 = BatchNormalization()(conv1)
            conv1 = Conv2D(filters=n_filters, kernel_size=(kernel_size,kernel_size),kernel_initializer='he_normal',padding='same')(conv1)
            conv1 = BatchNormalization()(conv1)
            conv1 = Activation('relu')(conv1)
            pool1 = MaxPooling2D(pool_size=(2,2))(conv1)

            conv2 = Conv2D(filters=n_filters*2, kernel_size=(kernel_size,kernel_size),kernel_initializer='he_normal',padding='same')(pool1)
            conv2 = BatchNormalization()(conv2)
            conv2 = Conv2D(filters=n_filters*2, kernel_size=(kernel_size,kernel_size),kernel_initializer='he_normal',padding='same')(conv2)
            conv2 = BatchNormalization()(conv2)
            conv2 = Activation('relu')(conv2)
            pool2 = MaxPooling2D(pool_size=(2,2))(conv2)

            conv3 = Conv2D(filters=n_filters*4, kernel_size=(kernel_size,kernel_size),kernel_initializer='he_normal',padding='same')(pool2)
            conv3 = BatchNormalization()(conv3)
            conv3 = Conv2D(filters=n_filters*4, kernel_size=(kernel_size,kernel_size),kernel_initializer='he_normal',padding='same')(conv3)
            conv3 = BatchNormalization()(conv3)
            conv3 = Activation('relu')(conv3)
            pool3 = MaxPooling2D(pool_size=(2,2))(conv3)

            conv4 = Conv2D(filters=n_filters*8, kernel_size=(kernel_size,kernel_size),kernel_initializer='he_normal',padding='same')(pool3)
            conv4 = BatchNormalization()(conv4)
            conv4 = Conv2D(filters=n_filters*8, kernel_size=(kernel_size,kernel_size),kernel_initializer='he_normal',padding='same')(conv4)
            conv4 = BatchNormalization()(conv4)
            conv4 = Activation('relu')(conv4)
            pool4 = MaxPooling2D(pool_size=(2,2))(conv4)

            conv5 = Conv2D(filters=n_filters*16, kernel_size=(kernel_size,kernel_size),kernel_initializer='he_normal',padding='same')(pool4)
            conv5 = BatchNormalization()(conv5)
            conv5 = Conv2D(filters=n_filters*16, kernel_size=(kernel_size,kernel_size),kernel_initializer='he_normal',padding='same')(conv5)
            conv5 = BatchNormalization()(conv5)
            conv5 = Activation('relu')(conv5)

            up6 = Conv2DTranspose(n_filters*8, (3,3),strides=(2,2), padding='same')(conv5)
            up6 = concatenate([up6,conv4])
            conv6 = Conv2D(filters=n_filters*8, kernel_size=(kernel_size,kernel_size),kernel_initializer='he_normal',padding='same')(up6)
            conv6 = BatchNormalization()(conv6)
            conv6 = Conv2D(filters=n_filters*8, kernel_size=(kernel_size,kernel_size),kernel_initializer='he_normal',padding='same')(conv6)
            conv6 = BatchNormalization()(conv6)
            conv6 = Activation('relu')(conv6)

            up7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same')(conv6)
            up7 = concatenate([up7, conv3])
            conv7 = Conv2D(filters=n_filters*4, kernel_size=(kernel_size,kernel_size),kernel_initializer='he_normal',padding='same')(up7)
            conv7 = BatchNormalization()(conv7)
            conv7 = Conv2D(filters=n_filters*4, kernel_size=(kernel_size,kernel_size),kernel_initializer='he_normal',padding='same')(conv7)
            conv7 = BatchNormalization()(conv7)
            conv7 = Activation('relu')(conv7)

            up8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(conv7)
            up8 = concatenate([up8, conv2])
            conv8 = Conv2D(filters=n_filters*2, kernel_size=(kernel_size,kernel_size),kernel_initializer='he_normal',padding='same')(up8)
            conv8 = BatchNormalization()(conv8)
            conv8 = Conv2D(filters=n_filters*2, kernel_size=(kernel_size,kernel_size),kernel_initializer='he_normal',padding='same')(conv8)
            conv8 = BatchNormalization()(conv8)
            conv8 = Activation('relu')(conv8)

            up9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(conv8)
            up9 = concatenate([up9, conv1])
            conv9 = Conv2D(filters=n_filters*1, kernel_size=(kernel_size,kernel_size),kernel_initializer='he_normal',padding='same')(up9)
            conv9 = BatchNormalization()(conv9)
            conv9 = Conv2D(filters=n_filters*1, kernel_size=(kernel_size,kernel_size),kernel_initializer='he_normal',padding='same')(conv9)
            conv9 = BatchNormalization()(conv9)
            conv9 = Activation('relu')(conv9)

            outputs = Conv2D(1 , (1,1), activation='sigmoid')(conv9)


            self.model = Model(input=inputs, output=outputs)

            self.model.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy')

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


    def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
        x =Conv2D(filters=n_filters, kernel_size=(kernel_size,kernel_size),kernel_initializer='he_normal',padding='same')(input_tensor)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters=n_filters, kernel_size=(kernel_size,kernel_size),kernel_initializer='he_normal',padding='same')(x)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x
