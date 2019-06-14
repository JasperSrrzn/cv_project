from keras.layers import *
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import *
import os
from keras import regularizers

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

    x_shortcut_1 = Conv2D(filters=n_filters, kernel_size=(kernel_size,kernel_size),kernel_initializer='he_normal',
              padding='same')(x)

    x =Conv2D(filters=n_filters, kernel_size=(kernel_size,kernel_size),kernel_initializer='he_normal',
              padding='same')(x)
    if batchnorm:
        x = BatchNormalization()(x)

    x = Activation('relu')(x)

    x = Conv2D(filters=n_filters, kernel_size=(kernel_size,kernel_size),kernel_initializer='he_normal',
              padding='same')(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Add()([x,x_shortcut_1])
    x = Activation('relu')(x)

    return x

class ConvolutionalAutoencoder(object):
    """
    constructs an object to easily call a convolutional autoencoder
    """
    def __init__(self,latent_dim,n_filters,pretrained_weights=None):
        """
        initializes the architecture of the model
        """
        self.latent_dim = latent_dim
        self.n_filters = n_filters

        if pretrained_weights==None:
            input_size = (224, 224, 3)
            inputs = Input(input_size)
            x = Conv2D(16, (3, 3), padding='same')(inputs)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = MaxPooling2D((2, 2), padding='same')(x)
            x = Conv2D(32, (3, 3), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = MaxPooling2D((2, 2), padding='same')(x)
            x = Conv2D(64, (3, 3), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = MaxPooling2D((2, 2), padding='same')(x)
            x = Conv2D(128, (3, 3), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            encoded = MaxPooling2D((2, 2), padding='same')(x)

            x = Conv2D(128, (3, 3), padding='same')(encoded)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = UpSampling2D((2, 2))(x)
            x = Conv2D(64, (3, 3), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = UpSampling2D((2, 2))(x)
            x = Conv2D(32, (3, 3), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = UpSampling2D((2, 2))(x)
            x = Conv2D(16, (3, 3), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = UpSampling2D((2, 2))(x)
            x = Conv2D(, (3, 3), padding='same')(x)
            x = BatchNormalization()(x)
            decoded = Activation('sigmoid')(x)


            self.encoder = Model(inputs, encoded)
            self.autoencoder = Model(input=inputs, output=decoded)

            self.autoencoder.compile(optimizer=Adam(lr=1e-3), loss='mean_squared_error')
            if (pretrained_weights):
                self.autoencoder.load_weights(pretrained_weights)




    def fit(self,X_train,X_validation,name,epochs=50):
        """
        function to fit the model constructed in the def __init__ function.
        """
        model_dir = '/content/gdrive/My Drive/autoencoders/saved_models/'
        callbacks = [ModelCheckpoint(model_dir+name,monitor='val_loss', verbose=1, save_best_only=True,
                                    save_weights_only=False),
                    EarlyStopping(patience=20, verbose=1),
                    ReduceLROnPlateau(patience=10, verbose=1),
                    TrainValTensorBoard(log_dir='/content/gdrive/My Drive/autoencoders/logs/'+name[:-3],
                                histogram_freq=0, batch_size=32, write_graph=True, write_grads=False,
                                write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                                embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
                                ]
        self.autoencoder.fit(x=X_train,y=X_train,epochs=epochs,
                            validation_data=[X_validation,X_validation],callbacks=callbacks)

    def save_weights(self,path=None,prefix=""):
        """
        function to save weight parameters of the model
        """
        if path is None:
            path = os.get.cwd()
        self.encoder.save_weights(os.path.join(path,prefix + "encoder_weights.h5"))
        self.decoder.save_weights(os.path.join(path_prefix + "decoder_weights.h5"))

    def load_weights(self,path=None):
        self.autoencoder.load_weights(path)

    def encode(self,input):
        """
        function to encode input
        """
        return self.encoder.predict(input)

    def decode(self,codes):
        """
        function to decode latent representations
        """
        return self.decoder.predict(codes)

    def predict(self,X):
        """
        function to reconstruct input images
        """
        return self.autoencoder.predict(X)

    def get_weights(self):
        """
        function to retrieve the weight parameters of the model
        """
        return self.autoencoder.get_weights()
