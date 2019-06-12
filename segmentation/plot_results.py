import numpy as np
import matplotlib.pyplot as plt
import os

def dice_coef(y_true, y_pred, smooth=1):
    intersection = np.sum(np.abs(y_true * y_pred))
    return (2. * intersection + smooth) / (np.sum(np.square(y_true)) + np.sum(np.square(y_pred)) + smooth)

data_dir = os.path.dirname(os.getcwd())+'/output_data/'
X_test = np.load(data_dir+'x_test_img.npy')
Y_test = np.load(data_dir+'y_test_seg.npy')
Y_test = np.reshape(Y_test,(Y_test.shape[0],Y_test.shape[1],Y_test.shape[2],1))

predictions_test_dice = np.load('test_segmentations_dice.npy')
predictions_test_xent = np.load('test_segmentations_xent.npy')
"""
id = 3
image1 = X_test[id]
pred_xent = predictions_test_xent[id,:,:,0]
pred_xent = np.reshape(pred_xent,(pred_xent.shape[0],pred_xent.shape[1],1))
segm_xent = np.concatenate((pred_xent,pred_xent,pred_xent),axis=-1)
pred_dice = predictions_test_dice[id,:,:,0]
pred_dice = np.reshape(pred_dice,(pred_dice.shape[0],pred_dice.shape[1],1))
segm_dice = np.concatenate((pred_dice,pred_dice,pred_dice),axis=-1)
plt.subplot(1,3,1)
plt.imshow(image1)
plt.subplot(1,3,2)
plt.imshow(segm_xent)
plt.subplot(1,3,3)
plt.imshow(segm_dice)
plt.show()
"""

print(dice_coef(Y_test[0,:,:,0],predictions_test_xent[0,:,:,0]))
