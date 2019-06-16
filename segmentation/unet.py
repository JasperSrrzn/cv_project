import os
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras import backend as K
import tensorflow as tf


class TrainValTensorBoard(TensorBoard):
    """
    tensorboard to plot both validation and training curves on one figure
    """
    def __init__(self, log_dir='./logs', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()

def dice_coef(y_true, y_pred, smooth=1):
    """
    calculates the dice coefficient
    """
    intersection = K.sum(K.abs(y_true * y_pred))
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),) + K.sum(K.square(y_pred)) + smooth)

def dice_coef_loss(y_true, y_pred):
    """
    calculates the dice loss
    """
    return 1-dice_coef(y_true, y_pred)


def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    """
    constructs a block of ResNet convolutional layers and batchnormalization layers
    """
    x_shortcut = Conv2D(filters=n_filters, kernel_size=(kernel_size,kernel_size),kernel_initializer='he_normal',
              padding='same')(input_tensor)
    x =Conv2D(filters=n_filters, kernel_size=(kernel_size,kernel_size),kernel_initializer='he_normal',
              padding='same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)

    x = Activation('relu')(x)

    x = Conv2D(filters=n_filters, kernel_size=(kernel_size,kernel_size),kernel_initializer='he_normal',
              padding='same')(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Add()([x,x_shortcut])
    x = Activation('relu')(x)

    return x


class Unet(object):
    """
    constructs an object to easily call a U-net
    """
    def __init__(self,n_filters,pretrained_weights=None):
        """
        initializes the architecture of the model
        """
        if pretrained_weights==None:
            input_size = (224, 224, 3)
            inputs = Input(input_size)
            conv1 = Conv2D(16, (3, 3), padding='same')(inputs)
            conv1 = BatchNormalization()(conv1)
            conv1 = Activation('relu')(conv1)
            pool1 = MaxPooling2D((2, 2), padding='same')(conv1)

            conv2 = Conv2D(32, (3, 3), padding='same')(pool1)
            conv2 = BatchNormalization()(conv2)
            conv2 = Activation('relu')(conv2)
            pool2 = MaxPooling2D((2, 2), padding='same')(conv2)


            conv3 = Conv2D(64, (3, 3), padding='same')(pool2)
            conv3 = BatchNormalization()(conv3)
            conv3 = Activation('relu')(conv3)
            pool3 = MaxPooling2D((2, 2), padding='same')(conv3)

            conv4 = Conv2D(64, (3, 3), padding='same')(pool3)
            conv4 = BatchNormalization()(conv4)
            conv4 = Activation('relu')(conv4)
            pool4 = MaxPooling2D((2, 2), padding='same')(conv4)

            conv5 = Conv2D(128, (3, 3), padding='same')(pool4)
            conv5 = BatchNormalization()(conv5)
            conv5 = Activation('relu')(conv5)

            encoded = MaxPooling2D((2, 2), padding='same')(conv5)

            up6 = UpSampling2D((2,2))(encoded)
            up6 = concatenate([up6, conv5])
            conv6 = Conv2D(128, (3,3), padding='same')(up6)
            conv6 = BatchNormalization()(conv6)
            conv6 = Activation('relu')(conv6)

            up7 = UpSampling2D((2,2))(conv6)
            up7 = concatenate([up7, conv4])
            conv7 = Conv2D(64, (3,3), padding='same')(up7)
            conv7 = BatchNormalization()(conv7)
            conv7 = Activation('relu')(conv7)

            up8 = UpSampling2D((2,2))(conv7)
            up8 = concatenate([up8, conv3])
            conv8 = Conv2D(32, (3,3), padding='same')(up8)
            conv8 = BatchNormalization()(conv8)
            conv8 = Activation('relu')(conv8)

            up9 = UpSampling2D((2,2))(conv8)
            up9 = concatenate([up9, conv2])
            conv9 = Conv2D(16, (3,3), padding='same')(up9)
            conv9 = BatchNormalization()(conv9)
            conv9 = Activation('relu')(conv9)

            up10 = UpSampling2D((2,2))(conv9)
            outputs = Conv2D(1,(1,1),padding='same',activation='sigmoid')(up10)

            self.model = Model(input=inputs, output=outputs)

            self.model.compile(optimizer=Adam(lr=1e-3), loss=dice_coef_loss)

            if (pretrained_weights):
                self.model.load_weights(pretrained_weights)

    def fit(self,X_train,y_train,X_validation,y_validation,name,epochs=50):
        """
        function to fit the model constructed in the def __init__ function.
        """
        model_dir = '/content/gdrive/My Drive/segmentation/saved_models/'
        callbacks = [ModelCheckpoint(model_dir+name,monitor='val_loss', verbose=1, save_best_only=True,
                                    save_weights_only=False),
                    EarlyStopping(patience=20, verbose=1),
                    ReduceLROnPlateau(patience=10, verbose=1),
                    TrainValTensorBoard(log_dir='/content/gdrive/My Drive/segmentation/logs/'+name[:-3],
                                histogram_freq=0, batch_size=32, write_graph=True, write_grads=False,
                                write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                                embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
                                ]
        self.model.fit(x=X_train, y=y_train, epochs=epochs,
                       validation_data=[X_validation, y_validation],callbacks=callbacks)

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
        self.model.fit_generator(datagen.flow(x=X_train, y=y_train, batch_size=16), steps_per_epoch=100, epochs=epochs,validation_data=[X_validation, y_validation],
                       callbacks=callbacks)


    def predict(self,X):
        """
        function to reconstruct input images
        """
        return self.model.predict(X)


    def load_weights(self,path=None):
        """
        function to load weight parameters in the model architecture
        """
        self.model.load_weights(path)
