import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Model, Sequential, Input, load_model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Flatten, Lambda
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, SGD, adagrad
from keras.initializers import RandomNormal
from keras.preprocessing.image import ImageDataGenerator

# from resnet152 import ResNet152

GPU = True
if GPU:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.7
    tf.keras.backend.set_session(tf.Session(config=config))


def create_base_model(image_shape, dropout_rate, suffix=''):
    I1 = Input(shape=image_shape)
    model = ResNet50(include_top=False, weights='imagenet', input_tensor=I1, pooling=None)
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []

    for layer in model.layers:
        layer.name = layer.name + str(suffix)
        layer.trainable = False

    flatten_name = 'flatten' + str(suffix)

    x = model.output
    x = Flatten(name=flatten_name)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(dropout_rate)(x)

    return x, model.input


def create_siamese_model(image_shape, dropout_rate):

    output_left, input_left = create_base_model(image_shape, dropout_rate)
    output_right, input_right = create_base_model(image_shape, dropout_rate, suffix="_2")

    L1_layer = Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([output_left, output_right])
    L1_prediction = Dense(1, use_bias=True,
                          activation='sigmoid',
                          kernel_initializer=RandomNormal(mean=0.0, stddev=0.001),
                          name='weighted-average')(L1_distance)

    # prediction = Dropout(0.2)(L1_prediction)

    siamese_model = Model(inputs=[input_left, input_right], outputs=L1_prediction)

    return siamese_model


siamese_model = create_siamese_model(image_shape=(224, 224, 3), dropout_rate=0.2)

siamese_model.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=0.0001),
                      metrics=['binary_crossentropy', 'acc'])
siamese_model.fit_generator(train_gen,
                            steps_per_epoch=1000,
                            epochs=10,
                            verbose=1,
                            callbacks=[checkpoint, tensor_board_callback, lr_reducer, early_stopper, csv_logger],
                            validation_data=validation_data,
                            max_q_size=3)

siamese_model.save('siamese_model.h5')



# and the my prediction
siamese_net = load_model('siamese_model.h5', custom_objects={"tf": tf})

X_1 = [image, ] * len(markers)
batch = [markers, X_1]
result = siamese_net.predict_on_batch(batch)

# I've tried also to check identical images
markers = [image]
X_1 = [image, ] * len(markers)
batch = [markers, X_1]
result = siamese_net.predict_on_batch(batch)