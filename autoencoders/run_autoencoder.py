from convolutional_autoencoder import ConvolutionalAutoencoder
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

# flag for training or read from test.h5
do_training = 1
save_autoenco = 1

#define parameters

num_epochs = 1000
num_filters = 4
latent_dimension = 6300
loss = 'mse'
print(latent_dimension)
#directory for stored data and stored model
data_dir = os.path.dirname(os.getcwd())+'/output_data/'
model_dir = '/content/gdrive/My Drive//autoencoders/saved_models/'

#name for storing best_autoencoder
name = 'autoencoder_'+loss+'_'+str(latent_dimension)+'.h5'


# read train,validation and test data
X_train = np.load(data_dir+'x_train_img.npy')
X_validation = np.load(data_dir+'x_val_img.npy')
X_test = np.load(data_dir+'x_test_img.npy')


if do_training == 1:
    print('starting training...')
    #initialize model
    autoencoder = ConvolutionalAutoencoder(latent_dimension,num_filters)
    #train model
    autoencoder.fit(X_train=X_train,X_validation=X_validation,name=name,epochs=num_epochs)

#select the best model (stored)
best_autoencoder = ConvolutionalAutoencoder(latent_dimension,num_filters)
best_autoencoder.load_weights(model_dir+name)


#encode the training set and test set
X_train_enc = best_autoencoder.encode(X_train)
X_test_enc = best_autoencoder.encode(X_test)
X_val_enc = best_autoencoder.encode(X_validation)

#reconstruct
X_test_rec = best_autoencoder.predict(X_test)

np.save('/content/gdrive/My Drive/autoencoders/reconstructions/X_pred_rec_'+name[:-3]+'.npy',X_test_rec)


# save encoded vars
if save_autoenco == 1:
    print('saving encoded vars')
    np.save('/content/gdrive/My Drive/autoencoders/encoded/X_train_enc_'+name[:-3]+'.npy',X_train_enc)
    np.save('/content/gdrive/My Drive/autoencoders/encoded/X_test_enc_'+name[:-3]+'.npy',X_test_enc)
    np.save('/content/gdrive/My Drive/autoencoders/encoded/X_val_enc_'+name[:-3]+'.npy',X_val_enc)
