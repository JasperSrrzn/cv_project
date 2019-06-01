from classifier import classifier
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

n_classes = 5
num_epochs = 1000
latent_dimension = 100
n_filters = 16
loss_of_autoencoder = 'xent'
name = 'classifier_'+loss_of_autoencoder+'_'+str(latent_dimension)+'.h5'
data_dir = os.path.dirname(os.getcwd())+'/output_data/'
model_dir_freeze = os.path.dirname(os.getcwd())+'/classifiers/saved_models/freeze/'
model_dir_unfreeze = os.path.dirname(os.getcwd())+'/classifiers/saved_models/unfreeze/'
model_dir_random = os.path.dirname(os.getcwd())+'/classifiers/saved_models/random/'

# read train data
X_train = np.load(data_dir+'x_train_img.npy')
Y_train = np.load(data_dir+'y_train_lab.npy')
X_validation = np.load(data_dir+'x_val_img.npy')
Y_validation = np.load(data_dir+'y_val_lab.npy')
X_test = np.load(data_dir+'x_test_img.npy')
Y_test = np.load(data_dir+'y_test_lab.npy')

X_train = np.array([X_train[i] for i in range(len(Y_train)) if sum(Y_train[i])==1])
Y_train = np.array([Y_train[i] for i in range(len(Y_train)) if sum(Y_train[i])==1])
X_validation = np.array([X_validation[i] for i in range(len(Y_validation)) if sum(Y_validation[i])==1])
Y_validation = np.array([Y_validation[i] for i in range(len(Y_validation)) if sum(Y_validation[i])==1])
X_test = np.array([X_test[i] for i in range(len(Y_test)) if sum(Y_test[i])==1])
Y_test = np.array([Y_test[i] for i in range(len(Y_test)) if sum(Y_test[i])==1])
#freezed
clf = classifier(latent_dimension,loss_of_autoencoder,n_filters)
clf.fit_freeze(X_train,Y_train,X_validation,Y_validation,num_epochs)
best_clf = classifier(latent_dimension,loss_of_autoencoder,n_filters)
best_clf.load_weights(model_dir_freeze+name)
Y_pred = best_clf.predict(X_test)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], Y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(Y_test.ravel(), Y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure()
lw = 2
plt.plot(fpr['micro'], tpr['micro'], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc['micro'])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.savefig('./figures/freeze/'+name[:-3]+'_roc.eps')

#unfreezed
clf = classifier(latent_dimension,loss_of_autoencoder,n_filters)
clf.fit_unfreeze(X_train,Y_train,X_validation,Y_validation,num_epochs)
best_clf = classifier(latent_dimension,loss_of_autoencoder,n_filters)
best_clf.load_weights(model_dir_unfreeze+name)
Y_pred = best_clf.predict(X_test)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], Y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(Y_test.ravel(), Y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure()
lw = 2
plt.plot(fpr['micro'], tpr['micro'], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc['micro'])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.savefig('./figures/unfreeze/'+name[:-3]+'_roc.eps')


#random unfreezed
clf = classifier(latent_dimension,loss_of_autoencoder,n_filters)
clf.fit_random(X_train,Y_train,X_validation,Y_validation,num_epochs)
best_clf = classifier(latent_dimension,loss_of_autoencoder,n_filters)
best_clf.load_weights(model_dir_random+name)
Y_pred = best_clf.predict(X_test)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], Y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(Y_test.ravel(), Y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure()
lw = 2
plt.plot(fpr['micro'], tpr['micro'], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc['micro'])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.savefig('./figures/random/'+name[:-3]+'_roc.eps')
