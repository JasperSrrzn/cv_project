import pandas as pd
import matplotlib.pyplot as plt

tr = pd.read_csv('run-training-tag-loss.csv')
val = pd.read_csv('run-validation-tag-loss.csv')

x = tr['Step']
y_tr = tr['Value']
y_val = val['Value']

plt.plot(x,y_val,label='validation loss')
plt.plot(x,y_tr,label='training loss')
plt.xlabel('epoch')
plt.ylabel('binary crossentropy')
plt.grid()
plt.savefig('train_val_loss.eps')
