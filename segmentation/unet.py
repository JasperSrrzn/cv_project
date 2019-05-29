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
    def __init__(self,pretrained_weights=None):
        if pretrained_weights==None:
            input_size = (224, 224, 3)
            inputs = Input(input_size)
            conv1 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
            conv1 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
            conv1 = BatchNormalization()(conv1)
            pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
            conv2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
            conv2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
            conv2 = BatchNormalization()(conv2)
            pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
            conv3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
            conv3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
            conv3 = BatchNormalization()(conv3)
            pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
            conv4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
            conv4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
            conv4 = BatchNormalization()(conv4)
            drop4 = Dropout(0.5)(conv4)
            pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
            conv5 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
            conv5 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
            conv5 = BatchNormalization()(conv5)
            drop5 = Dropout(0.5)(conv5)

            up6 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(drop5))
            merge6 = concatenate([drop4, up6], axis=3)
            conv6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
            conv6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
            conv6 = BatchNormalization()(conv6)

            up7 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv6))
            merge7 = concatenate([conv3, up7], axis=3)
            conv7 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
            conv7 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
            conv7 = BatchNormalization()(conv7)

            up8 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv7))
            merge8 = concatenate([conv2, up8], axis=3)
            conv8 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
            conv8 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
            conv8 = BatchNormalization()(conv8)
            up9 = Conv2D(16, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv8))
            merge9 = concatenate([conv1, up9], axis=3)
            conv9 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
            conv9 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
            conv9 = BatchNormalization()(conv9)
            conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
            conv9 = BatchNormalization()(conv9)
            conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

            self.model = Model(input=inputs, output=conv10)

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