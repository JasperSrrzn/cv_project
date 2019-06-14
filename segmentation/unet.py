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
            conv1 = conv2d_block(inputs, n_filters*1, kernel_size=3, batchnorm=True)
            pool1 = MaxPooling2D(pool_size=(2,2))(conv1)
            pool1 = Dropout(0.2)(pool1)

            conv2 = conv2d_block(pool1, n_filters*2, kernel_size=3, batchnorm=True)
            pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
            pool2 = Dropout(0.2)(pool2)

            conv3 = conv2d_block(pool2, n_filters*4, kernel_size=3, batchnorm=True)
            pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
            pool3 = Dropout(0.2)(pool3)

            conv4 = conv2d_block(pool3, n_filters*8, kernel_size=3, batchnorm=True)
            pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
            pool4 = Dropout(0.2)(pool4)

            conv5 = conv2d_block(pool4, n_filters*16, kernel_size=3, batchnorm=True)

            up6 = Conv2DTranspose(n_filters*8, (3,3),strides=(2,2), padding='same')(conv5)
            up6 = concatenate([up6,conv4])
            up6 = Dropout(0.2)(up6)
            conv6 = conv2d_block(up6, n_filters*8, kernel_size=3, batchnorm=True)

            up7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same')(conv6)
            up7 = concatenate([up7, conv3])
            up7 = Dropout(0.2)(up7)
            conv7 = conv2d_block(up7, n_filters * 4, kernel_size=3, batchnorm=True)

            up8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(conv7)
            up8 = concatenate([up8, conv2])
            up8 = Dropout(0.2)(up8)
            conv8 = conv2d_block(up8, n_filters * 2, kernel_size=3, batchnorm=True)

            up9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(conv8)
            up9 = concatenate([up9, conv1])
            up9  = Dropout(0.2)(up9)
            conv9 = conv2d_block(up9, n_filters * 1, kernel_size=3, batchnorm=True)

            outputs = Conv2D(1 , (1,1), activation='sigmoid')(conv9)

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
