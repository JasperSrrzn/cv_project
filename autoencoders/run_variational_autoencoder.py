from variational_autoencoder import VariationalConvolutionalAutoencoder
import os
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#define parameters
latent_dimension = 1000
num_epochs = 100

#directory for stored data and stored model
data_dir = os.path.dirname(os.getcwd())+'/output_data/'
model_dir = os.path.dirname(os.getcwd())+'/autoencoders/saved_models/'

#name for storing best_autoencoder
name = 'variationalautoencoder_xent_'+str(latent_dimension)+'.h5'

# read train data
X = np.load(data_dir+'X_train.npy')
Y = np.load(data_dir+'Y_train.npy')

#split training data in train and validation set
X_train, X_validation, Y_train, Y_validation = train_test_split(X,Y,test_size=0.2,random_state=42)

Y_train = np.argmax(Y_train,axis=1)+1
Y_validation = np.argmax(Y_validation,axis=1)+1

#read test data
X_test = np.load(data_dir+'X_test.npy')
Y_test = np.load(data_dir+'Y_test.npy')
Y_test = np.argmax(Y_test,axis=1)+1


vae = VariationalConvolutionalAutoencoder(latent_dimension)
vae.fit(X_train,X_validation,name,epochs=num_epochs)

best_vae = VariationalConvolutionalAutoencoder(latent_dimension)
best_vae.load_weights(model_dir+name)

X_train_enc = best_vae.encode(X_train)
X_test_enc = best_vae.encode(X_test)
X_test_rec = best_vae.decode(X_test_enc)
X_train_rec = best_vae.decode(X_train_enc)
print(mean_squared_error(X_test.flatten(),X_test_rec.flatten()))

#make tSNE plots to visualize the distribution of data points in 2d
train_codes_embedded = TSNE(n_components=2).fit_transform(X_train_enc)
test_codes_embedded = TSNE(n_components=2).fit_transform(X_test_enc)
c_dict = {1: 'red', 2:'blue',3:'green'}

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
try:
    os.mkdir('./figures/'+name[:-3])
except:
    pass
plt.savefig('./figures/'+name[:-3]+'/tsne_train.eps')


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
plt.savefig('./figures/'+name[:-3]+'/tsne_test.eps')

