import numpy as np
import os
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

#define parameters
latent_dimension1 = 1000
loss1 = 'xent'
latent_dimension2 = 1000
loss2 = 'mse'

#name for storing best_autoencoder
name1 = 'autoencoder_'+loss1+'_'+str(latent_dimension1)
name2 = 'autoencoder_'+loss2+'_'+str(latent_dimension2)

data_dir = os.path.dirname(os.path.dirname(os.getcwd()))+'/output_data/'
fig_dir = os.path.dirname(os.getcwd())+'/figures/'
enc_dir = os.path.dirname(os.getcwd())+'/encoded/'

Y_train = np.load(data_dir+'y_train_lab.npy')
Y_validation = np.load(data_dir+'y_val_lab.npy')
Y_test = np.load(data_dir+'y_test_lab.npy')
Y_train = np.argmax(Y_train,axis=1)+1
Y_validation = np.argmax(Y_validation,axis=1)+1
Y_test = np.argmax(Y_test,axis=1)+1


X_test = np.load(data_dir+'x_test_img.npy')
X_pred1 = np.load('X_pred_rec_'+name1+'.npy')
X_pred2 = np.load('X_pred_rec_'+name2+'.npy')


################make tSNE plots to visualize the distribution of data points in 2d #######################"
X_test_enc = np.load(enc_dir+'X_test_enc_'+name1+'.npy')
X_train_enc = np.load(enc_dir+'X_train_enc_'+name1+'.npy')
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
    os.mkdir(fig_dir+name1)
except:
    pass
plt.savefig(fig_dir+name1+'/tsne_test.eps')

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
plt.savefig(fig_dir+name1+'/tsne_train.eps')

#############plot example reconstruction ########################"
"""
test_id = 6
plt.imshow(X_test[test_id])
plt.savefig('original.eps')


test1 = X_pred1[test_id]
plt.imshow(test1)
plt.savefig('xent_reconstruction_1000.eps')

test2 = X_pred2[test_id]
plt.imshow(test2)
plt.savefig('mse_reconstruction_1000.eps')

############ calculate MSE on test ######################"

print(mean_squared_error(X_test.flatten(),X_pred1.flatten()))
"""
