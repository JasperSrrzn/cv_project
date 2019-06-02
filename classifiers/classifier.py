from convolutional_autoencoder import ConvolutionalAutoencoder
from keras.layers import Activation, Dense, BatchNormalization
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import os
import numpy as np
from keras import regularizers

class classifier(object):

    def __init__(self,latent_dimension,loss,n_filters):
        self.latent_dimension = latent_dimension
        self.loss = loss
        self.n_filters = n_filters
        self.name = 'classifier_'+loss+'_'+str(latent_dimension)+'.h5'
        self.ae_name = 'autoencoder_'+loss+'_'+str(latent_dimension)+'.h5'
        self.ae_model_dir = os.path.dirname(os.getcwd()) + '/autoencoders/saved_models/'
        self.ae = ConvolutionalAutoencoder(self.latent_dimension,self.n_filters)
        self.ae.load_weights(self.ae_model_dir+self.ae_name)
        self.encoder = self.ae.encoder
        x = BatchNormalization()(self.encoder.layers[-1].output)
        prediction = Dense(5,activation='softmax',kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(l2=0.001))(x)
        self.classifier = Model(input=self.encoder.input,output=prediction)
        self.classifier.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])


    def fit_freeze(self,X_train,y_train,X_validation,y_validation,epochs):
        for layer in self.classifier.layers[:-2]:
            layer.trainable = False
        model_dir = os.path.dirname(os.getcwd()) + '/classifiers/saved_models/freeze/'
        callbacks = [ModelCheckpoint(model_dir + self.name, monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=False),
                     EarlyStopping(patience=30, verbose=1),
                     ReduceLROnPlateau(patience=15, verbose=1),
                     TensorBoard(log_dir='./logs/freeze/' + self.name[:-3], histogram_freq=0, batch_size=32, write_graph=True,
                                 write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                                 embeddings_metadata=None, embeddings_data=None, update_freq='epoch')]

        self.classifier.fit(x=X_train, y=y_train,epochs=epochs, validation_data=[X_validation, y_validation],
                             callbacks=callbacks)

    def fit_unfreeze(self,X_train,y_train,X_validation,y_validation,epochs):
        model_dir = os.path.dirname(os.getcwd()) + '/classifiers/saved_models/unfreeze/'
        callbacks = [ModelCheckpoint(model_dir + self.name, monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=False),
                     EarlyStopping(patience=30, verbose=1),
                     ReduceLROnPlateau(patience=15, verbose=1),
                     TensorBoard(log_dir='./logs/unfreeze/' + self.name[:-3], histogram_freq=0, batch_size=32, write_graph=True,
                                 write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                                 embeddings_metadata=None, embeddings_data=None, update_freq='epoch')]
        self.classifier.fit(x=X_train, y=y_train,epochs=epochs, validation_data=[X_validation, y_validation],
                             callbacks=callbacks)


    def fit_random(self,X_train,y_train,X_validation,y_validation,epochs):
        self.reset_weights()
        model_dir = os.path.dirname(os.getcwd()) + '/classifiers/saved_models/random/'
        callbacks = [ModelCheckpoint(model_dir + self.name, monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=False),
                     EarlyStopping(patience=30, verbose=1),
                     ReduceLROnPlateau(patience=15, verbose=1),
                     TensorBoard(log_dir='./logs/unfreeze/' + self.name[:-3], histogram_freq=0, batch_size=32, write_graph=True,
                                 write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                                 embeddings_metadata=None, embeddings_data=None, update_freq='epoch')]
        self.classifier.fit(x=X_train, y=y_train,epochs=epochs, validation_data=[X_validation, y_validation],
                             callbacks=callbacks)

    def reset_weights(self):
        for layer in self.classifier.layers:
            if hasattr(layer,'kernel_initializer'):
                weights, biases  = layer.get_weights()
                weights_random = np.random.normal(0,0.05,weights.shape)
                biases_random = np.random.normal(0,0.05,biases.shape)
                layer.set_weights([weights_random,biases_random])


    def predict(self,X):
        return self.classifier.predict(X)

    def load_weights(self,path=None):
        self.classifier.load_weights(path)
