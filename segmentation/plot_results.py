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

predictions_test_dice = np.load('test_segmentations_dice_old.npy')
predictions_test_dice_new = np.load('test_segmentations_dice_new.npy')

id = 7
image1 = X_test[id]
pred_dice_new = predictions_test_dice_new[id,:,:,0]
pred_dice_new = np.reshape(pred_dice_new,(pred_dice_new.shape[0],pred_dice_new.shape[1],1))
segm_dice_new = np.concatenate((pred_dice_new,pred_dice_new,pred_dice_new),axis=-1)
pred_dice = predictions_test_dice[id,:,:,0]
pred_dice = np.reshape(pred_dice,(pred_dice.shape[0],pred_dice.shape[1],1))
segm_dice = np.concatenate((pred_dice,pred_dice,pred_dice),axis=-1)

plt.imshow(image1)
plt.savefig('original.eps')
plt.imshow(segm_dice_new)
plt.savefig('dice.eps')
plt.imshow(segm_dice)
plt.savefig('dice_new.eps')

test_set_scores = []
for i in range(len(Y_test)):
    test_set_scores.append(performance_score(Y_test[i,:,:,0],predictions_test_dice_new[i,:,:,0],0))
print(np.mean(test_set_scores))

test_set_scores = []
for i in range(len(Y_test)):
    test_set_scores.append(performance_score(Y_test[i,:,:,0],predictions_test_dice[i,:,:,0],0))
print(np.mean(test_set_scores))
