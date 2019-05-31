from convolutional_autoencoder import ConvolutionalAutoencoder
import os
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# flag for training or read from test.h5
do_training = 1
save_autoenco = 0

#define parameters
latent_dimension = 10
num_epochs = 100
num_filters = 16
loss = 'xent'

#directory for stored data and stored model
data_dir = os.path.dirname(os.getcwd())+'/output_data/'
model_dir = os.path.dirname(os.getcwd())+'/autoencoders/saved_models/'

#name for storing best_autoencoder
name = 'autoencoder_'+loss+'_'+str(latent_dimension)+'.h5'


# read train,validation and test data
X_train = np.load(data_dir+'x_train_img.npy')
X_validation = np.load(data_dir+'x_val_img.npy')
X_test = np.load(data_dir+'x_test_img.npy')


if do_training == 1:
    print('starting training')
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

np.save('/content/gdrive/My Drive/testreconstructions.npy',X_test_rec)

"""
# save encoded vars
if save_autoenco == 1:
    print('saving encoded vars')
    np.save(os.path.join(data_dir,'X_train_enc.npy'),X_train_enc)
    np.save(os.path.join(data_dir,'X_test_enc.npy'),X_test_enc)
    np.save(os.path.join(data_dir,'X_val_enc.npy'),X_val_enc)
    print('saving labels for training and validation')
    np.save(os.path.join(data_dir,'Y_train_train.npy'),Y_train)
    np.save(os.path.join(data_dir,'Y_train_val.npy'),Y_validation)
    np.save(os.path.join(data_dir,'Y_test_test.npy'),Y_test)



#make tSNE plots to visualize the distribution of data points in 2d
test_codes_embedded = TSNE(n_components=2).fit_transform(X_test_enc)
train_codes_embedded = TSNE(n_components=2).fit_transform(X_train_enc)
c_dict = {1: 'red', 2:'blue',3:'green',4:'yellow',5:'black'}
fig, ax = plt.subplots()
legend_labels = []
for g in range(len(test_codes_embedded)):
    if Y_test[g] not in legend_labels:
        plt.scatter(test_codes_embedded[g,0], test_codes_embedded[g,1], c=c_dict[Y_test[g]], label=Y_test[g])
        legend_labels += [Y_test[g]]
    else:
        plt.scatter(test_codes_embedded[g, 0], test_codes_embedded[g, 1], c=c_dict[Y_test[g]])
plt.legend()
plt.title('tSNE on test set')
try:
    os.mkdir('./figures/'+name[:-3])
except:
    pass
plt.savefig('./figures/'+name[:-3]+'/tsne_test.eps')

fig, ax = plt.subplots()
legend_labels = []
for g in range(len(train_codes_embedded)):
    if Y_train[g] not in legend_labels:
        plt.scatter(train_codes_embedded[g,0], train_codes_embedded[g,1], c=c_dict[Y_train[g]], label=Y_train[g])
        legend_labels += [Y_train[g]]
    else:
        plt.scatter(train_codes_embedded[g, 0], train_codes_embedded[g, 1], c=c_dict[Y_train[g]])
plt.legend()
plt.title('tSNE on training set')
plt.savefig('./figures/'+name[:-3]+'/tsne_train.eps')
"""
