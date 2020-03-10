import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import random

from sklearn.model_selection import train_test_split

from keras import backend as K
from keras.utils import to_categorical
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Model, Sequential, Input
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Flatten, Lambda
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, SGD, adagrad
from keras.preprocessing.image import ImageDataGenerator

# from resnet152 import ResNet152

def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    for d in range(num_classes):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)


def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


learning_rate = 0.0001
batch_size = 16
dense_layer = 0
dropout = 0
epochs = 50
optimizer = 'adam'

# imgDir = '/home/image/projects/pacesystem_data/siamese_data'
imgDir = 'D:/Freelancer/Fashion_Pacesystem/pacesystem_data/siamese_data'
model_name = f'result/{dense_layer}_{dropout}_{optimizer}_{epochs}_{learning_rate}_{batch_size}_TL_aug'
if os.path.isdir(model_name) is False:
    os.mkdir(model_name)
    os.mkdir(model_name + '/graph')

IMG_WIDTH = 224
IMG_HEIGHT = 224
IMG_DIM = (IMG_WIDTH, IMG_HEIGHT)

name_count = []
dirs = []
train_names = []
train_labels = []
train_dirs = sorted(os.listdir(f'{imgDir}/train'))
for i, dir in enumerate(train_dirs):
    names = glob.glob(f'{imgDir}/train/{dir}/*.jpg')
    name_count.append(len(names))
    dirs.append(dir)
    train_names += names
    train_labels += [i for _ in range(len(names))]
print(name_count)
print(dirs)
test_names = []
test_labels = []
test_dirs = sorted(os.listdir(f'{imgDir}/val'))
for i, dir in enumerate(test_dirs):
    names = glob.glob(f'{imgDir}/val/{dir}/*.jpg')
    test_names += names
    test_labels += [i for _ in range(len(names))]

temp = list(zip(train_names, train_labels))
random.shuffle(temp)
train_names, train_labels = zip(*temp)

# temp = list(zip(test_names, test_labels))
# random.shuffle(temp)
# test_names, test_labels = zip(*temp)

num_classes = len(set(train_labels))
print(set(train_labels))
print(f'num_classes : {num_classes}')

train_x, test_x = train_names, test_names
train_y, test_y = train_labels, test_labels

train_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in train_x]
train_imgs = np.array(train_imgs)
train_labels = np.array(train_y)

test_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in test_x]
test_imgs = np.array(test_imgs)
test_labels = np.array(test_y)

print(f'Train dataset shape:{train_imgs.shape}')
print(f'Test dataset shape:{test_imgs.shape}')

X_train = train_imgs.astype('float32') / 255.
X_test = test_imgs.astype('float32') / 255.
Y_train = train_labels
Y_test = test_labels

digit_indices = [np.where(Y_train == i)[0] for i in range(num_classes)]
tr_pairs, tr_y = create_pairs(X_train, digit_indices)

digit_indices = [np.where(Y_test == i)[0] for i in range(num_classes)]
te_pairs, te_y = create_pairs(X_test, digit_indices)

##################################################################################################

input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)

left_input = Input(input_shape)
right_input = Input(input_shape)

resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = resnet.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dense(num_classes, activation='sigmoid')(x)
model = Model(inputs=resnet.input, outputs=x)

encoded_l = model(left_input)
encoded_r = model(right_input)

Euc_layer = Lambda(lambda tensor: K.abs(tensor[0] - tensor[1]))
Euc_distance = Euc_layer([encoded_l, encoded_r])

prediction = Dense(1, activation='sigmoid')(Euc_distance)
siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)

# distance = Lambda(euclidean_distance,
#                   output_shape=eucl_dist_output_shape)([encoded_l, encoded_r])
# distance = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
# distance = distance([encoded_l, encoded_r])
# siamese_net = Model([left_input, right_input], distance)
# prediction = Dense(1, activation='sigmoid', bias_initializer=ini)

# opt = SGD(lr=learning_rate, decay=.01, momentum=0.9, nesterov=True)
opt = Adam(lr=learning_rate)
siamese_net.compile(loss='binary_crossentropy', optimizer=optimizer)
# model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
# siamese_net.compile(optimizer=opt, loss=contrastive_loss, metrics=['accuracy'])
filepath = model_name + "/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

##################################################################################################


###########################################################################################

import tensorflow as tf
import os
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
siamese_net.summary()

# with tf.device('/gpu:0'):
history = siamese_net.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
                          batch_size=batch_size,
                          epochs=epochs,
                          callbacks=callbacks_list,
                          verbose=1,
                          validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))

y_pred = siamese_net.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
tr_acc = compute_accuracy(tr_y, y_pred)
y_pred = siamese_net.predict([te_pairs[:, 0], te_pairs[:, 1]])
te_acc = compute_accuracy(te_y, y_pred)

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

# summarize history for accuracy
plot_name = model_name + '/graph/'
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(f'{plot_name}accuracy.png')

# summarize history for loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(f'{plot_name}loss.png')
preds = model.evaluate(X_test, Y_test)
print("Loss = " + str(preds[0]))
print("Test Accuracy = " + str(preds[1]))
