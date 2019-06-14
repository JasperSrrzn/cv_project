from keras.layers import *
from keras.models import Model, Sequential
from keras import backend as K
from keras.losses import binary_crossentropy, mse
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import os
from keras import regularizers
from keras.optimizers import Adam
import tensorflow as tf

def sampling(args):
    """
    function to draw samples in the latent space
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5*z_log_var)*epsilon

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

class VariationalConvolutionalAutoencoder(object):
    """
    function to create an object for the Variational Autoencoder
    """
    def __init__(self,latent_dim,n_filters,pretrained_weights=None):

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

            #generate latent representation

            m = BatchNormalization()(latent)
            z_mean = Dense(int(shape[1])*int(shape[2])*int(shape[3]),kernel_initializer='he_normal')(m)


            l = BatchNormalization()(latent)
            z_logvar = Dense(int(shape[1])*int(shape[2])*int(shape[3]),kernel_initializer='he_normal')(l)

            sample = Lambda(sampling)([z_mean, z_logvar])
            #sample = sample_layer

            self.encoder = Model(input=inputs,output=[z_mean,z_logvar])

            f2 = Reshape((int(shape[1]),int(shape[2]),int(shape[3])))(sample)

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
            self.vae = Model(inputs,outputs)
            reconstruction_loss =  mse(K.flatten(inputs), K.flatten(outputs))
            kl_loss = 1 + z_logvar - K.square(z_mean) - K.exp(z_logvar)
            kl_loss = K.sum(kl_loss, axis=-1)
            kl_loss *= -0.5
            vae_loss = K.mean(reconstruction_loss + kl_loss)
            self.vae.add_loss(vae_loss)
            adam = Adam(lr=1e-3)
            self.vae.compile(optimizer=adam)

    def fit(self,X_train,X_validation,name,epochs=50):
        """
        function to fit the variational autoencoder
        """
        model_dir = '/content/gdrive/My Drive/autoencoders/saved_models/'
        callbacks = [ModelCheckpoint(model_dir + name, monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=False),
                     EarlyStopping(patience=20, verbose=1),
                     ReduceLROnPlateau(patience=10, verbose=1),
                     TrainValTensorBoard(log_dir='/content/gdrive/My Drive/autoencoders/logs/'+name[:-3],
                                 histogram_freq=0, batch_size=32, write_graph=True, write_grads=False,
                                 write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                                 embeddings_metadata=None, embeddings_data=None, update_freq='epoch')]
        self.vae.fit(x=X_train,y=None,epochs=epochs,validation_data=[X_validation,None],callbacks=callbacks)

    def encode(self,X):
        """
        function to encode the data
        """
        return self.encoder.predict(X)

    def decode(self,Z):
        """
        function to decode the latent representation
        """
        return self.decoder.predict(Z)

    def load_weights(self,path=None):
        """
        function to load the weight parameters in a model
        """
        self.vae.load_weights(path)

    def predict(self,X):
        """
        make reconstruction
        """
        return self.vae.predict(X)

    def get_weights(self):
        """
        get the weight parameters of the model
        """
        return self.vae.get_weights()
