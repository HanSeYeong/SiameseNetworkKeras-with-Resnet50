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

GPU = True
if GPU:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.7
    tf.keras.backend.set_session(tf.Session(config=config))

learning_rate = 0.001
batch_size = 16
dense_layer = 0
dropout = 0
epochs = 30
optimizer = 'adam'

imgDir = '/home/image/projects/pacesystem_data/siamese_data'
model_name = f'result/{dense_layer}_{dropout}_{optimizer}_{epochs}_{learning_rate}_{batch_size}_TL_aug'
if os.path.isdir(model_name) is False:
    os.mkdir(model_name)
    os.mkdir(model_name + '/graph')

IMG_WIDTH = 224
IMG_HEIGHT = 224
IMG_DIM = (IMG_WIDTH, IMG_HEIGHT)

train_names = []
train_labels = []
train_dirs = sorted(os.listdir(f'{imgDir}/train'))
for i, dir in enumerate(train_dirs):
    names = glob.glob(f'{imgDir}/train/{dir}/*.jpg')
    train_names += names
    train_labels += [i for _ in range(len(names))]

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

###################################################################################################

input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)

left_input = Input(input_shape)
right_input = Input(input_shape)

resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = resnet.output
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(num_classes, activation='sigmoid')(x)
model = Model(inputs=resnet.input, outputs=x)

encoded_l = model(left_input)
encoded_r = model(right_input)

Euc_layer = Lambda(lambda tensor: K.abs(tensor[0] - tensor[1]))
Euc_distance = Euc_layer([encoded_l, encoded_r])

prediction = Dense(1, activation='sigmoid')(Euc_distance)
siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)

opt = SGD(lr=learning_rate, decay=.01, momentum=0.9, nesterov=True)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
filepath = model_name + "/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

##################################################################################################

number_of_train = len(X_test)
image_list = X_train
label_list = Y_train

left_input = []
right_input = []
targets = []

pairs = num_classes

for i in range(number_of_train):
    for j in range(pairs):
        compare_to = i
        while compare_to == i:
            compare_to = random.randint(0, number_of_train)
        left_input.append(image_list[i])
        right_input.append(image_list[compare_to])
        if label_list[i] == label_list[compare_to]:
            targets.append(1.)
        else:
            targets.append(0.)

left_input = np.squeeze(np.array(left_input))
right_input = np.squeeze(np.array(right_input))
targets = np.squeeze(np.array(targets))

test_left = []
test_right = []
test_targets = []

for i in range(len(Y_test)):
    test_left.append(X_train[0])
    test_right.append(X_test)
    test_targets.append(Y_test)

test_left = np.squeeze(np.array(test_left))
test_right = np.squeeze(np.array(test_right))
test_targets = np.squeeze(np.array(test_targets))

###########################################################################################

import tensorflow as tf
import os
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
siamese_net.summary()

with tf.device('/gpu:1'):
    history = siamese_net.fit([left_input, right_input], targets,
                              batch_size=batch_size,
                              epochs=epochs,
                              verbose=1,
                              validation_data=([test_left, test_right], test_targets))


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
