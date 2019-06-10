from keras.layers import *
from keras.models import Model, Sequential
from keras import backend as K
from keras.losses import binary_crossentropy, mse
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import os
from keras import regularizers
from keras.optimizers import Adam

def sampling(args):
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
    constructs a block of convolutional layers and batchnormalization layers
    """
    x =Conv2D(filters=n_filters, kernel_size=(kernel_size,kernel_size),kernel_initializer='he_normal',
              padding='same',kernel_regularizer=regularizers.l2(0.01))(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=n_filters, kernel_size=(kernel_size,kernel_size),kernel_initializer='he_normal',
              padding='same',kernel_regularizer=regularizers.l2(0.01))(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x

class VariationalConvolutionalAutoencoder(object):

    def __init__(self,latent_dim,n_filters,pretrained_weights=None):

        self.latent_dim = latent_dim
        self.n_filters = n_filters

        if pretrained_weights==None:
            input_size = (224, 224, 3)
            inputs = Input(input_size)
            conv1 = conv2d_block(inputs, n_filters*1, kernel_size=3, batchnorm=True)
            self.enc1 = Model(input=inputs,output=conv1)
            pool1 = MaxPooling2D(pool_size=(2,2))(conv1)
            pool1 = Dropout(0.2)(pool1)

            conv2 = conv2d_block(pool1, n_filters*2, kernel_size=3, batchnorm=True)
            self.enc2 = Model(input=inputs,output=conv2)
            pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
            pool2 = Dropout(0.2)(pool2)

            conv3 = conv2d_block(pool2, n_filters*4, kernel_size=3, batchnorm=True)
            self.enc3 = Model(input=inputs,output=conv3)
            pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
            pool3 = Dropout(0.2)(pool3)

            conv4 = conv2d_block(pool3, n_filters*8, kernel_size=3, batchnorm=True)
            self.enc4 = Model(input=inputs,output=conv4)
            pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
            pool4 = Dropout(0.2)(pool4)

            conv5 = conv2d_block(pool4, n_filters*16, kernel_size=3, batchnorm=True)

            shape = conv5.shape
            latent = Flatten()(conv5)

            #generate latent representation

            m = BatchNormalization()(latent)
            z_mean = Dense(self.latent_dim)(m)


            l = BatchNormalization()(latent)
            z_logvar = Dense(self.latent_dim)(l)

            sample = Lambda(sampling)([z_mean, z_logvar])
            #sample = sample_layer

            input1_size = conv1.shape
            input2_size = conv2.shape
            input3_size = conv3.shape
            input4_size = conv4.shape
            inputs1 = Input(input1_size)(conv1)
            inputs2 = Input(input2_size)(conv2)
            inputs3 = Input(input3_size)(conv3)
            inputs4 = Input(input4_size)(conv4)

            self.encoder = Model(input=inputs,output=[z_mean,z_logvar,conv1,conv2,conv3,conv4])

            f2 = Dense((int(shape[1])*int(shape[2])*int(shape[3])))(sample)
            f2 = Reshape((int(shape[1]),int(shape[2]),int(shape[3])))(f2)

            up6 = Conv2DTranspose(n_filters*8, (3,3),strides=(2,2), padding='same')(f2)
            up6 = concatenate([up6,inputs4])
            up6 = Dropout(0.2)(up6)
            conv6 = conv2d_block(up6, n_filters*8, kernel_size=3, batchnorm=True)

            up7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same')(conv6)
            up7 = concatenate([up7, inputs3])
            up7 = Dropout(0.2)(up7)
            conv7 = conv2d_block(up7, n_filters * 4, kernel_size=3, batchnorm=True)

            up8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(conv7)
            up8 = concatenate([up8, inputs2])
            up8 = Dropout(0.2)(up8)
            conv8 = conv2d_block(up8, n_filters * 2, kernel_size=3, batchnorm=True)

            up9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(conv8)
            up9 = concatenate([up9, inputs1])
            up9  = Dropout(0.2)(up9)
            conv9 = conv2d_block(up9, n_filters * 1, kernel_size=3, batchnorm=True)

            outputs = Conv2D(3 , (1,1), activation='sigmoid')(conv9)
            self.decoder = Model(input=[sample,inputs1,inputs2,inputs3,inputs4],output=ouputs)
            self.vae = Model(inputs,outputs)
            reconstruction_loss =  binary_crossentropy(K.flatten(inputs), K.flatten(outputs))
            kl_loss = 1 + z_logvar - K.square(z_mean) - K.exp(z_logvar)
            kl_loss = K.sum(kl_loss, axis=-1)
            kl_loss *= -0.5
            vae_loss = K.mean(reconstruction_loss + kl_loss)
            self.vae.add_loss(vae_loss)
            adam = Adam(lr=1e-4)
            self.vae.compile(optimizer=adam)

    def fit(self,X_train,X_validation,name,epochs=50):
        model_dir = '/content/gdrive/My Drive/autoencoders/saved_models/'
        callbacks = [ModelCheckpoint(model_dir + name, monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=False),
                     EarlyStopping(patience=20, verbose=1),
                     ReduceLROnPlateau(patience=10, verbose=1),
                     TensorBoard(log_dir='/content/gdrive/My Drive/autoencoders/logs/'+name[:-3], histogram_freq=0, batch_size=32, write_graph=True,
                                 write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                                 embeddings_metadata=None, embeddings_data=None, update_freq='epoch')]
        self.vae.fit(x=X_train,y=None,epochs=epochs,validation_data=[X_validation,None],callbacks=callbacks)

    def encode(self,X):
        return self.encoder.predict(X)

    def decode(self,Z):
        return self.decoder.predict(Z,conv1,conv2,conv3,conv4)

    def load_weights(self,path=None):
        self.vae.load_weights(path)

    def predict(self,X):
        return self.vae.predict(X)

    def get_weights(self):
        return self.vae.get_weights()
