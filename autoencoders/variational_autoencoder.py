from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, Input,InputLayer, Dense, Flatten, Reshape, Lambda, Conv2DTranspose
from keras.models import Model, Sequential
from keras import backend as K
from keras.losses import binary_crossentropy, mse
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import os

def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5*z_log_var)*epsilon

class VariationalConvolutionalAutoencoder(object):

    def __init__(self,latent_dim):
        self.latent_dim = latent_dim

        # encoder architecture
        self.encoder = Sequential()
        self.encoder.add(InputLayer((224, 224, 3)))
        self.encoder.add(Conv2D(4, (3, 3), padding='same'))
        self.encoder.add(BatchNormalization())
        self.encoder.add(Activation('relu'))
        self.encoder.add(MaxPooling2D((2, 2), padding='same'))
        self.encoder.add(Conv2D(8, (3, 3), padding='same'))
        self.encoder.add(BatchNormalization())
        self.encoder.add(Activation('relu'))
        self.encoder.add(MaxPooling2D((2, 2), padding='same'))
        self.encoder.add(Conv2D(16, (3, 3), padding='same'))
        self.encoder.add(BatchNormalization())
        self.encoder.add(Activation('relu'))
        self.encoder.add(MaxPooling2D((2, 2), padding='same'))
        self.encoder.add(Conv2D(32, (3, 3), padding='same'))
        self.encoder.add(BatchNormalization())
        self.encoder.add(Activation('relu'))
        self.encoder.add(MaxPooling2D((2, 2), padding='same'))
        self.encoder.add(Conv2D(64, (3, 3), padding='same'))
        self.encoder.add(BatchNormalization())
        self.encoder.add(Activation('relu'))
        self.encoder.add(MaxPooling2D((2, 2), padding='same'))
        shape = self.encoder.output_shape
        self.encoder.add(Flatten())
        flattened_shape = self.encoder.output_shape

        #generate latent representation
        self.mean_encoder = Sequential()
        self.mean_encoder.add(Dense(flattened_shape[-1]))
        self.mean_encoder.add(Activation('relu'))
        self.mean_encoder.add(BatchNormalization())
        self.mean_encoder.add(Dense(self.latent_dim))


        self.logvar_encoder = Sequential()
        self.logvar_encoder.add(Dense(flattened_shape[-1]))
        self.logvar_encoder.add(Activation('relu'))
        self.logvar_encoder.add(BatchNormalization())
        self.logvar_encoder.add(Dense(self.latent_dim))
        self.logvar_encoder.add(Dense(self.latent_dim))


        input = Input((224, 224, 3))
        code = self.encoder(input)
        z_mean = self.mean_encoder(code)
        z_logvar = self.logvar_encoder(code)
        sample_layer = Lambda(sampling)
        sample = sample_layer([z_mean, z_logvar])

        self.latent_encoder = Model(input,z_mean)

        self.decoder = Sequential()
        self.decoder.add(Dense(100))
        self.decoder.add(Activation('relu'))
        self.decoder.add(BatchNormalization())
        self.decoder.add(Dense(shape[1]*shape[2]*shape[3]))
        self.decoder.add(Reshape((shape[1],shape[2],shape[3])))
        self.decoder.add(Conv2DTranspose(64, (3, 3), padding='same'))
        self.decoder.add(BatchNormalization())
        self.decoder.add(Activation('relu'))
        self.decoder.add(UpSampling2D((2, 2)))
        self.decoder.add(Conv2DTranspose(32, (3, 3), padding='same'))
        self.decoder.add(BatchNormalization())
        self.decoder.add(Activation('relu'))
        self.decoder.add(UpSampling2D((2, 2)))
        self.decoder.add(Conv2DTranspose(16, (3, 3), padding='same'))
        self.decoder.add(BatchNormalization())
        self.decoder.add(Activation('relu'))
        self.decoder.add(UpSampling2D((2, 2)))
        self.decoder.add(Conv2DTranspose(8, (3, 3), padding='same'))
        self.decoder.add(BatchNormalization())
        self.decoder.add(Activation('relu'))
        self.decoder.add(UpSampling2D((2, 2)))
        self.decoder.add(Conv2DTranspose(4, (3, 3), padding='same'))
        self.decoder.add(BatchNormalization())
        self.decoder.add(Activation('relu'))
        self.decoder.add(UpSampling2D((2, 2)))
        self.decoder.add(Conv2DTranspose(3, (3, 3), padding='same'))
        self.decoder.add(BatchNormalization())
        self.decoder.add(Activation('sigmoid'))


        reconstructed = self.decoder(sample)

        self.vae = Model(input,reconstructed)
        reconstruction_loss =  binary_crossentropy(K.flatten(input), K.flatten(reconstructed))
        kl_loss = 1 + z_logvar - K.square(z_mean) - K.exp(z_logvar)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        self.vae.add_loss(vae_loss)
        self.vae.compile(optimizer='adam')

    def fit(self,X_train,X_validation,name,epochs=50):
        model_dir = os.path.dirname(os.getcwd()) + '/autoencoders/saved_models/'
        callbacks = [ModelCheckpoint(model_dir + name, monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=False),
                     EarlyStopping(patience=20, verbose=1),
                     ReduceLROnPlateau(patience=10, verbose=1),
                     TensorBoard(log_dir='./logs/'+name[:-3], histogram_freq=0, batch_size=32, write_graph=True,
                                 write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                                 embeddings_metadata=None, embeddings_data=None, update_freq='epoch')]
        self.vae.fit(x=X_train,epochs=epochs,batch_size=2,validation_data=(X_validation,None),callbacks=callbacks)

    def encode(self,X):
        return self.latent_encoder.predict(X)

    def decode(self,X):
        return self.decoder.predict(X)

    def load_weights(self,path=None):
        self.vae.load_weights(path)





