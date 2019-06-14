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
            conv1 = conv2d_block(inputs, n_filters*1, kernel_size=3, batchnorm=True)
            pool1 = MaxPooling2D(pool_size=(2,2))(conv1)

            conv2 = conv2d_block(pool1, n_filters*2, kernel_size=3, batchnorm=True)
            pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

            conv3 = conv2d_block(pool2, n_filters*4, kernel_size=3, batchnorm=True)
            pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

            conv4 = conv2d_block(pool3, n_filters*8, kernel_size=3, batchnorm=True)
            pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

            conv5 = conv2d_block(pool4, n_filters*16, kernel_size=3, batchnorm=True)
            pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

            conv6 = conv2d_block(pool5, n_filters*32, kernel_size=3, batchnorm=True)

            shape = conv6.shape
            latent = Flatten()(conv6)

            self.encoder = Model(input=inputs,output=latent)

            f2 = Reshape((int(shape[1]),int(shape[2]),int(shape[3])))(latent)

            up7 = Conv2DTranspose(n_filters*16, (3,3),strides=(2,2), padding='same',activation='relu')(f2)
            conv7 = conv2d_block(up7, n_filters*16, kernel_size=3, batchnorm=True)

            up8 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same',activation='relu')(conv7)
            conv8 = conv2d_block(up8, n_filters * 4, kernel_size=3, batchnorm=True)

            up9 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same',activation='relu')(conv8)
            conv9 = conv2d_block(up9, n_filters * 2, kernel_size=3, batchnorm=True)

            up10 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same',activation='relu')(conv9)
            conv10 = conv2d_block(up10, n_filters * 2, kernel_size=3, batchnorm=True)

            up11 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same',activation='relu')(conv10)
            conv11 = conv2d_block(up11, n_filters * 1, kernel_size=3, batchnorm=True)

            outputs = Conv2D(3 , (1,1), activation='sigmoid')(conv11)

            self.autoencoder = Model(input=inputs, output=outputs)

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

    def fit_generator(self,X_train,X_validation,datagen,name,epochs=50):
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
        datagen.fit(X_train)
        self.autoencoder.fit_generator(datagen.flow(x=X_train,y=X_train,batch_size=int(len(X_train)/9)), steps_per_epoch = 9*4, epochs=epochs,
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
