from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, Input,InputLayer, Conv2DTranspose, Dense, Flatten, Reshape
from keras.models import Model, Sequential
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import os

class fcn(object):
    def __init__(self,latent_dim):

        self.latent_dim = latent_dim

        #encoder architecture
        self.encoder = Sequential()
        self.encoder.add(InputLayer((224,224,3)))
        self.encoder.add(Conv2D(4,(3,3), padding='same'))
        self.encoder.add(BatchNormalization())
        self.encoder.add(Activation('relu'))
        self.encoder.add(MaxPooling2D((2,2),padding='same'))
        self.encoder.add(Conv2D(8,(3,3),padding='same'))
        self.encoder.add(BatchNormalization())
        self.encoder.add(Activation('relu'))
        self.encoder.add(MaxPooling2D((2,2),padding='same'))
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
        flatten_shape = self.encoder.output_shape
        self.encoder.add(Dense(flatten_shape[-1]))
        self.encoder.add(Activation('relu'))
        self.encoder.add(BatchNormalization())
        self.encoder.add(Dense(self.latent_dim))

        #decoder architecture
        self.decoder = Sequential()
        self.decoder.add(Dense(flatten_shape[-1]))
        self.decoder.add(Activation('relu'))
        self.decoder.add(BatchNormalization())
        self.decoder.add(Dense(shape[1] * shape[2] * shape[3]))
        self.decoder.add(Reshape((shape[1], shape[2], shape[3])))
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
        self.decoder.add(Conv2DTranspose(8,(3,3),padding='same'))
        self.decoder.add(BatchNormalization())
        self.decoder.add(Activation('relu'))
        self.decoder.add(UpSampling2D((2,2)))
        self.decoder.add(Conv2DTranspose(4,(3,3),padding='same'))
        self.decoder.add(BatchNormalization())
        self.decoder.add(Activation('relu'))
        self.decoder.add(UpSampling2D((2,2)))
        self.decoder.add(Conv2DTranspose(3,(3,3),padding='same'))
        self.decoder.add(BatchNormalization())
        self.decoder.add(Conv2DTranspose(1,(1,1),padding='same'))
        self.decoder.add(Activation('sigmoid'))

        input = Input((224,224,3))
        code = self.encoder(input)
        segmented = self.decoder(code)


        self.model = Model(input, segmented)
        self.model.compile(optimizer='adam', loss='binary_crossentropy')



    def fit(self,X_train,y_train,X_validation,y_validation,name,epochs=50):
        model_dir = os.path.dirname(os.getcwd()) + '/segmentation/saved_models/'
        callbacks = [ModelCheckpoint(model_dir+name,monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False),
                    EarlyStopping(patience=20, verbose=1),
                    ReduceLROnPlateau(patience=10, verbose=1),
                    TensorBoard(log_dir='./logs/'+name[:-3], histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')]
        self.model.fit(x=X_train,y=y_train,epochs=epochs,validation_data=[X_validation,y_validation],callbacks=callbacks)

    def save_weights(self,path=None,prefix=""):
        if path is None:
            path = os.get.cwd()
        self.encoder.save_weights(os.path.join(path,prefix + "encoder_weights.h5"))
        self.decoder.save_weights(os.path.join(path_prefix + "decoder_weights.h5"))

    def load_weights(self,path=None):
        self.model.load_weights(path)

    def encode(self,input):
        return self.encoder.predict(input)

    def decode(self,codes):
        return self.decoder.predict(codes)

    def predict(self,X):
        return self.model.predict(X)


