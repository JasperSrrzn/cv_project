import numpy as np
import matplotlib.pyplot as plt
import os

def dice_coef(y_true, y_pred, threshold, smooth=1):
    y_pred_h = y_pred>threshold
    y_pred = y_pred_h.astype(int)
    intersection = np.sum(np.abs(y_true * y_pred))
    return (2. * intersection + smooth) / (np.sum(np.square(y_true)) + np.sum(np.square(y_pred)) + smooth)


def performance_score(y_true,y_pred,smooth):
    scores = []
    thresholds = [0.5, 0.55, 0.65 , 0.70, 0.75 , 0.80, 0.85,0.90,0.95]
    for threshold in thresholds:
        scores.append(dice_coef(y_true,y_pred,threshold,smooth))
    return np.mean(scores)

data_dir = os.path.dirname(os.getcwd())+'/output_data/'
X_test = np.load(data_dir+'x_test_img.npy')
Y_test = np.load(data_dir+'y_test_seg.npy')
Y_test = np.reshape(Y_test,(Y_test.shape[0],Y_test.shape[1],Y_test.shape[2],1))

predictions_test_dice = np.load('test_segmentations_dice.npy')
predictions_test_xent = np.load('test_segmentations_xent.npy')
"""
id = 7
image1 = X_test[id]
pred_xent = predictions_test_xent[id,:,:,0]
pred_xent = np.reshape(pred_xent,(pred_xent.shape[0],pred_xent.shape[1],1))
segm_xent = np.concatenate((pred_xent,pred_xent,pred_xent),axis=-1)
pred_dice = predictions_test_dice[id,:,:,0]
pred_dice = np.reshape(pred_dice,(pred_dice.shape[0],pred_dice.shape[1],1))
segm_dice = np.concatenate((pred_dice,pred_dice,pred_dice),axis=-1)

plt.imshow(image1)
plt.savefig('original.eps')
plt.imshow(segm_xent)
plt.savefig('xent.eps')
plt.imshow(segm_dice)
plt.savefig('dice.jpg')
"""
test_set_scores = []
for i in range(len(Y_test)):
    test_set_scores.append(performance_score(Y_test[i,:,:,0],predictions_test_xent[i,:,:,0],0))
print(np.mean(test_set_scores))

test_set_scores = []
for i in range(len(Y_test)):
    test_set_scores.append(performance_score(Y_test[i,:,:,0],predictions_test_dice[i,:,:,0],0))
print(np.mean(test_set_scores))
