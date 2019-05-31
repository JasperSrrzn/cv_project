from keras.layers import *
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import os

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

class ConvolutionalAutoencoder(object):
    def __init__(self,latent_dim,n_filters,pretrained_weights=None):

        self.latent_dim = latent_dim
        self.n_filters = n_filters
        if pretrained_weights == None:
            input_size = (224, 224, 3)
            inputs = Input(input_size)
            conv1 = conv2d_block(inputs, self.n_filters * 1, kernel_size=3, batchnorm=True)
            pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
            pool1 = Dropout(0.2)(pool1)

            conv2 = conv2d_block(pool1, self.n_filters * 2, kernel_size=3, batchnorm=True)
            pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
            pool2 = Dropout(0.2)(pool2)

            conv3 = conv2d_block(pool2, self.n_filters * 4, kernel_size=3, batchnorm=True)
            pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
            pool3 = Dropout(0.2)(pool3)

            conv4 = conv2d_block(pool3, self.n_filters * 8, kernel_size=3, batchnorm=True)
            pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
            pool4 = Dropout(0.2)(pool4)

            conv5 = conv2d_block(pool4, self.n_filters * 16, kernel_size=3, batchnorm=True)
            shape = conv5.output_shape
            flatten = Flatten()(conv5)
            flatten_shape = flatten.shape

            f1 = Dense(flatten_shape[-1])(flatten)
            f1 = Activation('relu')(f1)
            f1 = BatchNormalization(f1)
            latent = Dense(self.latent_dim)(f1)

            self.encoder = Model(input=inputs,output=latent)

            f2 = Dense(flatten_shape[-1])(latent)
            f2 = Activation('relu')(f2)
            f2 = BatchNormalization()(f2)
            f2 = Dense(shape[1]*shape[2]*shape[3])(f2)
            f2 = Reshape((shape[1],shape[2],shape[3]))(f2)

            up6 = Conv2DTranspose(self.n_filters * 8, (3, 3), strides=(2, 2), padding='same')(f2)
            up6 = concatenate([up6, conv4])
            up6 = Dropout(0.2)(up6)
            conv6 = conv2d_block(up6, self.n_filters * 8, kernel_size=3, batchnorm=True)

            up7 = Conv2DTranspose(self.n_filters * 4, (3, 3), strides=(2, 2), padding='same')(conv6)
            up7 = concatenate([up7, conv3])
            up7 = Dropout(0.2)(up7)
            conv7 = conv2d_block(up7, self.n_filters * 4, kernel_size=3, batchnorm=True)

            up8 = Conv2DTranspose(self.n_filters * 2, (3, 3), strides=(2, 2), padding='same')(conv7)
            up8 = concatenate([up8, conv2])
            up8 = Dropout(0.2)(up8)
            conv8 = conv2d_block(up8, self.n_filters * 2, kernel_size=3, batchnorm=True)

            up9 = Conv2DTranspose(self.n_filters * 1, (3, 3), strides=(2, 2), padding='same')(conv8)
            up9 = concatenate([up9, conv1])
            up9 = Dropout(0.2)(up9)
            conv9 = conv2d_block(up9, self.n_filters * 1, kernel_size=3, batchnorm=True)

            outputs = Conv2D(3, (3, 3), activation='sigmoid')(conv9)

            self.decoder = Model(input=latent,output=outputs)
            self.autoencoder = Model(input=inputs, output=outputs)

            self.autoencoder.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy')

            if (pretrained_weights):
                self.model.load_weights(pretrained_weights)



    def fit(self,X_train,X_validation,name,epochs=50):
        model_dir = os.path.dirname(os.getcwd()) + '/content/gdrive/My Drive/autoencoders/saved_models/'
        callbacks = [ModelCheckpoint(model_dir+name,monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False),
                    EarlyStopping(patience=20, verbose=1),
                    ReduceLROnPlateau(patience=10, verbose=1),
                    TensorBoard(log_dir='/content/gdrive/My Drive/autoencoders/logs/'+name[:-3], histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')]
        self.autoencoder.fit(x=X_train,y=X_train,epochs=epochs,validation_data=[X_validation,X_validation],callbacks=callbacks)

    def save_weights(self,path=None,prefix=""):
        if path is None:
            path = os.get.cwd()
        self.encoder.save_weights(os.path.join(path,prefix + "encoder_weights.h5"))
        self.decoder.save_weights(os.path.join(path_prefix + "decoder_weights.h5"))

    def load_weights(self,path=None):
        self.autoencoder.load_weights(path)

    def encode(self,input):
        return self.encoder.predict(input)

    def decode(self,codes):
        return self.decoder.predict(codes)

    def predict(self,X):
        return self.autoencoder.predict(X)