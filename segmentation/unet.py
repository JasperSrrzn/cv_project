import os
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras import backend as K
import tensorflow as tf

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred))
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),) + K.sum(K.square(y_pred)) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def lovasz_grad(gt_sorted):
"""
Computes gradient of the Lovasz extension w.r.t sorted errors
See Alg. 1 in paper
"""
gts = tf.reduce_sum(gt_sorted)
intersection = gts - tf.cumsum(gt_sorted)
union = gts + tf.cumsum(1. - gt_sorted)
jaccard = 1. - intersection / union
jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
return jaccard


# --------------------------- BINARY LOSSES ---------------------------

def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        def treat_image(log_lab):
            log, lab = log_lab
            log, lab = tf.expand_dims(log, 0), tf.expand_dims(lab, 0)
            log, lab = flatten_binary_scores(log, lab, ignore)
            return lovasz_hinge_flat(log, lab)
        losses = tf.map_fn(treat_image, (logits, labels), dtype=tf.float32)
        loss = tf.reduce_mean(losses)
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """

    def compute_loss():
        labelsf = tf.cast(labels, logits.dtype)
        signs = 2. * labelsf - 1.
        errors = 1. - logits * tf.stop_gradient(signs)
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort")
        gt_sorted = tf.gather(labelsf, perm)
        grad = lovasz_grad(gt_sorted)
        loss = tf.tensordot(tf.nn.relu(errors_sorted), tf.stop_gradient(grad), 1, name="loss_non_void")
        return loss

    # deal with the void prediction case (only void pixels)
    loss = tf.cond(tf.equal(tf.shape(logits)[0], 0),
                   lambda: tf.reduce_sum(logits) * 0.,
                   compute_loss,
                   strict=True,
                   name="loss"
                   )
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = tf.reshape(scores, (-1,))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return scores, labels
    valid = tf.not_equal(labels, ignore)
    vscores = tf.boolean_mask(scores, valid, name='valid_scores')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vscores, vlabels

def lovasz_loss(y_true, y_pred):
    y_true, y_pred = K.cast(K.squeeze(y_true, -1), 'int32'), K.cast(K.squeeze(y_pred, -1), 'float32')
    #logits = K.log(y_pred / (1. - y_pred))
    logits = y_pred #Jiaxin
    loss = lovasz_hinge(logits, y_true, per_image = True, ignore = None)
    return loss


def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    x =Conv2D(filters=n_filters, kernel_size=(kernel_size,kernel_size),kernel_initializer='he_normal',padding='same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size,kernel_size),kernel_initializer='he_normal',padding='same')(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


class Unet(object):
    def __init__(self,n_filters,pretrained_weights=None):
        if pretrained_weights==None:
            input_size = (224, 224, 3)
            inputs = Input(input_size)
            conv1 = conv2d_block(inputs, n_filters*1, kernel_size=3, batchnorm=True)
            pool1 = MaxPooling2D(pool_size=(2,2))(conv1)
            pool1 = Dropout(0.2)(pool1)

            conv2 = conv2d_block(pool1, n_filters*2, kernel_size=3, batchnorm=True)
            pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
            pool2 = Dropout(0.2)(pool2)

            conv3 = conv2d_block(pool2, n_filters*4, kernel_size=3, batchnorm=True)
            pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
            pool3 = Dropout(0.2)(pool3)

            conv4 = conv2d_block(pool3, n_filters*8, kernel_size=3, batchnorm=True)
            pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
            pool4 = Dropout(0.2)(pool4)

            conv5 = conv2d_block(pool4, n_filters*16, kernel_size=3, batchnorm=True)

            up6 = Conv2DTranspose(n_filters*8, (3,3),strides=(2,2), padding='same')(conv5)
            up6 = concatenate([up6,conv4])
            up6 = Dropout(0.2)(up6)
            conv6 = conv2d_block(up6, n_filters*8, kernel_size=3, batchnorm=True)

            up7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same')(conv6)
            up7 = concatenate([up7, conv3])
            up7 = Dropout(0.2)(up7)
            conv7 = conv2d_block(up7, n_filters * 4, kernel_size=3, batchnorm=True)

            up8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(conv7)
            up8 = concatenate([up8, conv2])
            up8 = Dropout(0.2)(up8)
            conv8 = conv2d_block(up8, n_filters * 2, kernel_size=3, batchnorm=True)

            up9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(conv8)
            up9 = concatenate([up9, conv1])
            up9  = Dropout(0.2)(up9)
            conv9 = conv2d_block(up9, n_filters * 1, kernel_size=3, batchnorm=True)

            outputs = Conv2D(1 , (1,1), activation='sigmoid')(conv9)

            self.model = Model(input=inputs, output=outputs)

            self.model.compile(optimizer=Adam(lr=1e-3), loss=lovasz_loss)

            if (pretrained_weights):
                self.model.load_weights(pretrained_weights)

    def fit(self,X_train,y_train,X_validation,y_validation,name,epochs=50):
        model_dir = '/content/gdrive/My Drive/segmentation/saved_models/'
        callbacks = [ModelCheckpoint(model_dir + name, monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=False),
                     EarlyStopping(patience=20, verbose=1),
                     ReduceLROnPlateau(patience=10, verbose=1),
                     TensorBoard(log_dir='./logs/' + name[:-3], histogram_freq=0, batch_size=32, write_graph=True,
                                 write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                                 embeddings_metadata=None, embeddings_data=None, update_freq='epoch')]
        self.model.fit(x=X_train, y=y_train, epochs=epochs, validation_data=[X_validation, y_validation],
                       callbacks=callbacks)

    def fit_generator(self,X_train,y_train,X_validation,y_validation,datagen,name,epochs=50):
        model_dir = os.path.dirname(os.getcwd()) + '/segmentation/saved_models/'
        callbacks = [ModelCheckpoint(model_dir + name, monitor='val_loss', verbose=1, save_best_only=True,
                                     save_weights_only=False),
                     EarlyStopping(patience=20, verbose=1),
                     ReduceLROnPlateau(patience=10, verbose=1),
                     TensorBoard(log_dir='./logs/' + name[:-3], histogram_freq=0, batch_size=32, write_graph=True,
                                 write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                                 embeddings_metadata=None, embeddings_data=None, update_freq='epoch')]
        datagen.fit(X_train)
        self.model.fit_generator(datagen.flow(x=X_train, y=y_train, batch_size=16), steps_per_epoch=100, epochs=epochs,validation_data=[X_validation, y_validation],
                       callbacks=callbacks)

    def predict(self,X):
        return self.model.predict(X)


    def load_weights(self,path=None):
        self.model.load_weights(path)
