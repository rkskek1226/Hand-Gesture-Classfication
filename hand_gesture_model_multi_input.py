import numpy as np
import tensorflow as tf
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, concatenate, BatchNormalization, Activation, GlobalAveragePooling2D
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split, StratifiedKFold
from keras.datasets import mnist
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score
from sklearn import svm, datasets, tree
from keras.layers import Input
from keras.models import Model


import datetime
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
from keras.utils import np_utils
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Resizing, BatchNormalization, Activation, Dropout, GlobalAveragePooling2D, concatenate
from keras.models import Sequential
from google.colab import drive
from tensorflow.keras.layers.experimental.preprocessing import Resizing, Rescaling
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.layers import Input
from keras.models import Model
from tensorflow.keras.utils import plot_model


seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)

batch_size = 32
img_height = 480
img_width = 640

Data_set = pd.read_csv("/content/gdrive/MyDrive/Colab Notebooks/qwer.csv")
train_mlp_x_data = Data_set.iloc[:, 20:45].astype(float)
train_mlp_y_data = Data_set.iloc[:, 45]

train_mlp_y_data = np_utils.to_categorical(train_mlp_y_data)

# train_mlp_x_data, test_mlp_x_data, train_mlp_y_data, test_mlp_y_data = train_test_split(train_mlp_x_data, train_mlp_y_data, test_size=0.2, shuffle=True, stratify=train_mlp_y_data, random_state=1)
# train_mlp_x_data, test_mlp_x_data, train_mlp_y_data, test_mlp_y_data = train_test_split(train_mlp_x_data, train_mlp_y_data, shuffle=True, random_state=1)


# train_cnn_data = ImageDataGenerator(rescale=1. / 255, rotation_range=50, width_shift_range=0.6,
#                             height_shift_range=0.6, shear_range=0.6, zoom_range=0.6, horizontal_flip=True, vertical_flip=True, validation_split=0.2)

train_cnn_data = ImageDataGenerator(rescale=1. / 255, rotation_range=50, width_shift_range=0.6,
                            height_shift_range=0.6, shear_range=0.6, zoom_range=0.6, horizontal_flip=True, vertical_flip=True)


# train_cnn_data = train_cnn_data.flow_from_directory("/content/gdrive/MyDrive/Colab Notebooks/data2/", target_size=(img_height, img_width),
#                                             color_mode="rgb", batch_size=batch_size, seed=1, shuffle=True, class_mode="categorical", subset="training")
tmp = train_cnn_data.flow_from_directory("/content/gdrive/MyDrive/Colab Notebooks/data2/", target_size=(img_height, img_width),
                                            color_mode="rgb", batch_size=batch_size, seed=1, shuffle=False, class_mode="categorical")

# test_cnn_data = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)
# test_cnn_data = test_cnn_data.flow_from_directory("/content/gdrive/MyDrive/Colab Notebooks/data2/", target_size=(img_height, img_width),
#                                         color_mode="rgb", batch_size=batch_size, seed=2, shuffle=True,
#                                          class_mode="categorical", subset="validation")

# train_cnn_data = tf.keras.utils.image_dataset_from_directory("/content/gdrive/MyDrive/Colab Notebooks/data2/", labels="inferred", batch_size=batch_size, image_size=(img_height, img_width), shuffle=False, seed=1)

data_list = []
batch_index = 0

while batch_index <= tmp.batch_index:
    data = tmp.next()
    data_list.append(data[0])
    batch_index += 1


train_cnn_data = np.asarray(train_cnn_data)


# MLP~~~~~~~~~~~~~~~~~~~~
mlp_input = Input(shape=(25, ), batch_size=32, name="mlp_input")
mlp_hidden1 = Dense(64)(mlp_input)
mlp_hidden1 = BatchNormalization()(mlp_hidden1)
mlp_hidden1 = Activation("relu")(mlp_hidden1)
mlp_hidden1 = Dropout(0.25)(mlp_hidden1)

mlp_hidden2 = Dense(128)(mlp_hidden1)
mlp_hidden2 = BatchNormalization()(mlp_hidden2)
mlp_hidden2 = Activation("relu")(mlp_hidden2)
mlp_hidden2 = Dropout(0.25)(mlp_hidden2)

mlp_hidden3 = Dense(256)(mlp_hidden2)
mlp_hidden3 = BatchNormalization()(mlp_hidden3)
mlp_hidden3 = Activation("relu")(mlp_hidden3)
mlp_hidden3 = Dropout(0.25)(mlp_hidden3)

mlp_hidden4 = Dense(128)(mlp_hidden3)
mlp_hidden4 = BatchNormalization()(mlp_hidden4)
mlp_hidden4 = Activation("relu")(mlp_hidden4)
mlp_hidden4 = Dropout(0.25)(mlp_hidden4)

mlp_hidden5 = Dense(64)(mlp_hidden4)
mlp_hidden5 = BatchNormalization()(mlp_hidden5)
mlp_hidden5 = Activation("relu")(mlp_hidden5)
mlp_hidden5 = Dropout(0.25)(mlp_hidden5)

mlp_hidden6 = Dense(32)(mlp_hidden5)
mlp_hidden6 = BatchNormalization()(mlp_hidden6)
mlp_hidden6 = Activation("relu")(mlp_hidden6)
mlp_hidden6 = Dropout(0.25)(mlp_hidden6)

# mlp_output = Dense(10, activation="softmax")(mlp_hidden6)
mlp_output = Dense(40, name="mlp_output")(mlp_hidden6)

mlp_model = Model(inputs=mlp_input, outputs=mlp_output)


# CNN~~~~~~~~~~~~~~~~~~~~
cnn_input = tf.keras.Input((img_height, img_width, 3), name="cnn_input")
# conv_1
conv1 = Conv2D(64, (7, 7), padding="same", strides=2)(cnn_input)
conv1 = BatchNormalization()(conv1)
conv1 = Activation("relu")(conv1)
conv1 = MaxPooling2D((3, 3), padding="SAME", strides=2)(conv1)

# conv_2_x
conv2_1 = Conv2D(64, (3, 3), padding="same", strides=1)(conv1)
conv2_1 = BatchNormalization()(conv2_1)
conv2_1 = Activation("relu")(conv2_1)
conv2_1 = Dropout(0.5)(conv2_1)
conv2_1 = Conv2D(64, (3, 3), padding="same", strides=1)(conv2_1)
conv2_1 = BatchNormalization()(conv2_1)
conv2_1 = Activation("relu")(conv2_1)
conv2_1 = Dropout(0.5)(conv2_1)
short_cut = Conv2D(64, (1, 1), padding="same", strides=1)(conv1)
conv2_1 = tf.keras.layers.Add()([conv2_1, short_cut])
conv2_1 = BatchNormalization()(conv2_1)
conv2_1 = Activation("relu")(conv2_1)

conv2_2 = Conv2D(64, (3, 3), padding="same", strides=1)(conv2_1)
conv2_2 = BatchNormalization()(conv2_2)
conv2_2 = Activation("relu")(conv2_2)
conv2_2 = Dropout(0.5)(conv2_2)
conv2_2 = Conv2D(64, (3, 3), padding="same", strides=1)(conv2_2)
conv2_2 = BatchNormalization()(conv2_2)
conv2_2 = Activation("relu")(conv2_2)
conv2_2 = Dropout(0.5)(conv2_2)
conv2_2 = tf.keras.layers.Add()([conv2_2, conv2_1])
conv2_2 = BatchNormalization()(conv2_2)
conv2_2 = Activation("relu")(conv2_2)

conv2_3 = Conv2D(64, (3, 3), padding="same", strides=1)(conv2_2)
conv2_3 = BatchNormalization()(conv2_3)
conv2_3 = Activation("relu")(conv2_3)
conv2_3 = Dropout(0.5)(conv2_3)
conv2_3 = Conv2D(64, (3, 3), padding="same", strides=1)(conv2_3)
conv2_3 = BatchNormalization()(conv2_3)
conv2_3 = Activation("relu")(conv2_3)
conv2_3 = Dropout(0.5)(conv2_3)
conv2_3 = tf.keras.layers.Add()([conv2_3, conv2_2])
conv2_3 = BatchNormalization()(conv2_3)
conv2_3 = Activation("relu")(conv2_3)

# conv_3_x
conv3_1 = Conv2D(128, (3, 3), padding="same", strides=2)(conv2_3)
conv3_1 = BatchNormalization()(conv3_1)
conv3_1 = Activation("relu")(conv3_1)
conv3_1 = Dropout(0.5)(conv3_1)
conv3_1 = Conv2D(128, (3, 3), padding="same", strides=1)(conv3_1)
conv3_1 = BatchNormalization()(conv3_1)
conv3_1 = Activation("relu")(conv3_1)
conv3_1 = Dropout(0.5)(conv3_1)
short_cut = Conv2D(128, (1, 1), padding="same", strides=2)(conv2_3)
conv3_1 = tf.keras.layers.Add()([conv3_1, short_cut])
conv3_1 = BatchNormalization()(conv3_1)
conv3_1 = Activation("relu")(conv3_1)

conv3_2 = Conv2D(128, (3, 3), padding="same", strides=1)(conv3_1)
conv3_2 = BatchNormalization()(conv3_2)
conv3_2 = Activation("relu")(conv3_2)
conv3_2 = Dropout(0.5)(conv3_2)
conv3_2 = Conv2D(128, (3, 3), padding="same", strides=1)(conv3_2)
conv3_2 = BatchNormalization()(conv3_2)
conv3_2 = Activation("relu")(conv3_2)
conv3_2 = Dropout(0.5)(conv3_2)
conv3_2 = tf.keras.layers.Add()([conv3_2, conv3_1])
conv3_2 = BatchNormalization()(conv3_2)
conv3_2 = Activation("relu")(conv3_2)

conv3_3 = Conv2D(128, (3, 3), padding="same", strides=1)(conv3_2)
conv3_3 = BatchNormalization()(conv3_3)
conv3_3 = Activation("relu")(conv3_3)
conv3_3 = Dropout(0.5)(conv3_3)
conv3_3 = Conv2D(128, (3, 3), padding="same", strides=1)(conv3_3)
conv3_3 = BatchNormalization()(conv3_3)
conv3_3 = Activation("relu")(conv3_3)
conv3_3 = Dropout(0.5)(conv3_3)
conv3_3 = tf.keras.layers.Add()([conv3_3, conv3_2])
conv3_3 = BatchNormalization()(conv3_3)
conv3_3 = Activation("relu")(conv3_3)

conv3_4 = Conv2D(128, (3, 3), padding="same", strides=1)(conv3_3)
conv3_4 = BatchNormalization()(conv3_4)
conv3_4 = Activation("relu")(conv3_4)
conv3_4 = Dropout(0.5)(conv3_4)
conv3_4 = Conv2D(128, (3, 3), padding="same", strides=1)(conv3_4)
conv3_4 = BatchNormalization()(conv3_4)
conv3_4 = Activation("relu")(conv3_4)
conv3_4 = Dropout(0.5)(conv3_4)
conv3_4 = tf.keras.layers.Add()([conv3_4, conv3_3])
conv3_4 = BatchNormalization()(conv3_4)
conv3_4 = Activation("relu")(conv3_4)

# conv_4_x
conv4_1 = Conv2D(256, (3, 3), padding="same", strides=2)(conv3_4)
conv4_1 = BatchNormalization()(conv4_1)
conv4_1 = Activation("relu")(conv4_1)
conv4_1 = Dropout(0.5)(conv4_1)
conv4_1 = Conv2D(256, (3, 3), padding="same", strides=1)(conv4_1)
conv4_1 = BatchNormalization()(conv4_1)
conv4_1 = Activation("relu")(conv4_1)
conv4_1 = Dropout(0.5)(conv4_1)
short_cut = Conv2D(256, (1, 1), padding="same", strides=2)(conv3_4)
conv4_1 = tf.keras.layers.Add()([conv4_1, short_cut])
conv4_1 = BatchNormalization()(conv4_1)
conv4_1 = Activation("relu")(conv4_1)

conv4_2 = Conv2D(256, (3, 3), padding="same", strides=1)(conv4_1)
conv4_2 = BatchNormalization()(conv4_2)
conv4_2 = Activation("relu")(conv4_2)
conv4_2 = Dropout(0.5)(conv4_2)
conv4_2 = Conv2D(256, (3, 3), padding="same", strides=1)(conv4_2)
conv4_2 = BatchNormalization()(conv4_2)
conv4_2 = Activation("relu")(conv4_2)
conv4_2 = Dropout(0.5)(conv4_2)
conv4_2 = tf.keras.layers.Add()([conv4_2, conv4_1])
conv4_2 = BatchNormalization()(conv4_2)
conv4_2 = Activation("relu")(conv4_2)

conv4_3 = Conv2D(256, (3, 3), padding="same", strides=1)(conv4_2)
conv4_3 = BatchNormalization()(conv4_3)
conv4_3 = Activation("relu")(conv4_3)
conv4_3 = Dropout(0.5)(conv4_3)
conv4_3 = Conv2D(256, (3, 3), padding="same", strides=1)(conv4_3)
conv4_3 = BatchNormalization()(conv4_3)
conv4_3 = Activation("relu")(conv4_3)
conv4_3 = Dropout(0.5)(conv4_3)
conv4_3 = tf.keras.layers.Add()([conv4_3, conv4_2])
conv4_3 = BatchNormalization()(conv4_3)
conv4_3 = Activation("relu")(conv4_3)

conv4_4 = Conv2D(256, (3, 3), padding="same", strides=1)(conv4_3)
conv4_4 = BatchNormalization()(conv4_4)
conv4_4 = Activation("relu")(conv4_4)
conv4_4 = Dropout(0.5)(conv4_4)
conv4_4 = Conv2D(256, (3, 3), padding="same", strides=1)(conv4_4)
conv4_4 = BatchNormalization()(conv4_4)
conv4_4 = Activation("relu")(conv4_4)
conv4_4 = Dropout(0.5)(conv4_4)
conv4_4 = tf.keras.layers.Add()([conv4_4, conv4_3])
conv4_4 = BatchNormalization()(conv4_4)
conv4_4 = Activation("relu")(conv4_4)

conv4_5 = Conv2D(256, (3, 3), padding="same", strides=1)(conv4_4)
conv4_5 = BatchNormalization()(conv4_5)
conv4_5 = Activation("relu")(conv4_5)
conv4_5 = Dropout(0.5)(conv4_5)
conv4_5 = Conv2D(256, (3, 3), padding="same", strides=1)(conv4_5)
conv4_5 = BatchNormalization()(conv4_5)
conv4_5 = Activation("relu")(conv4_5)
conv4_5 = Dropout(0.5)(conv4_5)
conv4_5 = tf.keras.layers.Add()([conv4_5, conv4_4])
conv4_5 = BatchNormalization()(conv4_5)
conv4_5 = Activation("relu")(conv4_5)

conv4_6 = Conv2D(256, (3, 3), padding="same", strides=1)(conv4_5)
conv4_6 = BatchNormalization()(conv4_6)
conv4_6 = Activation("relu")(conv4_6)
conv4_6 = Dropout(0.5)(conv4_6)
conv4_6 = Conv2D(256, (3, 3), padding="same", strides=1)(conv4_6)
conv4_6 = BatchNormalization()(conv4_6)
conv4_6 = Activation("relu")(conv4_6)
conv4_6 = Dropout(0.5)(conv4_6)
conv4_6 = tf.keras.layers.Add()([conv4_6, conv4_5])
conv4_6 = BatchNormalization()(conv4_6)
conv4_6 = Activation("relu")(conv4_6)

# conv_5_x
conv5_1 = Conv2D(512, (3, 3), padding="same", strides=2)(conv4_6)
conv5_1 = BatchNormalization()(conv5_1)
conv5_1 = Activation("relu")(conv5_1)
conv5_1 = Dropout(0.5)(conv5_1)
conv5_1 = Conv2D(512, (3, 3), padding="same", strides=1)(conv5_1)
conv5_1 = BatchNormalization()(conv5_1)
conv5_1 = Activation("relu")(conv5_1)
conv5_1 = Dropout(0.5)(conv5_1)
short_cut = Conv2D(512, (1, 1), padding="same", strides=2)(conv4_6)
conv5_1 = tf.keras.layers.Add()([conv5_1, short_cut])
conv5_1 = BatchNormalization()(conv5_1)
conv5_1 = Activation("relu")(conv5_1)

conv5_2 = Conv2D(512, (3, 3), padding="same", strides=1)(conv5_1)
conv5_2 = BatchNormalization()(conv5_2)
conv5_2 = Activation("relu")(conv5_2)
conv5_2 = Dropout(0.5)(conv5_2)
conv5_2 = Conv2D(512, (3, 3), padding="same", strides=1)(conv5_2)
conv5_2 = BatchNormalization()(conv5_2)
conv5_2 = Activation("relu")(conv5_2)
conv5_2 = Dropout(0.5)(conv5_2)
conv5_2 = tf.keras.layers.Add()([conv5_2, conv5_1])
conv5_2 = BatchNormalization()(conv5_2)
conv5_2 = Activation("relu")(conv5_2)

conv5_3 = Conv2D(512, (3, 3), padding="same", strides=1)(conv5_2)
conv5_3 = BatchNormalization()(conv5_3)
conv5_3 = Activation("relu")(conv5_3)
conv5_3 = Dropout(0.5)(conv5_3)
conv5_3 = Conv2D(512, (3, 3), padding="same", strides=1)(conv5_3)
conv5_3 = BatchNormalization()(conv5_3)
conv5_3 = Activation("relu")(conv5_3)
conv5_3 = Dropout(0.5)(conv5_3)
conv5_3 = tf.keras.layers.Add()([conv5_3, conv5_2])
conv5_3 = BatchNormalization()(conv5_3)
conv5_3 = Activation("relu")(conv5_3)

avg_pool = GlobalAveragePooling2D()(conv5_3)
flat = Flatten()(avg_pool)
# cnn_output = Dense(10, activsation="softmax")(flat)
cnn_output = Dense(40, name="cnn_output")(flat)

cnn_model = Model(inputs=cnn_input, outputs=cnn_output)


concatenated = concatenate([mlp_output, cnn_output])
concatenated = Dense(40, activation="relu")(concatenated)
concat_output = Dense(10, activation="softmax")(concatenated)

model = Model(inputs=[mlp_input, cnn_input], outputs=concat_output)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


# train_mlp_x_data = np.asarray(train_mlp_x_data)
# train_cnn_data = np.asarray(train_cnn_data)
# rain_mlp_y_data = np.asarray(train_mlp_y_data)

print("train_mlp_x_data :", len(train_mlp_x_data))
print("train_mlp_x_data.shape :", train_mlp_x_data.shape)
print("train_mlp_y_data :", len(train_mlp_y_data))
print("train_mlp_y_data.shape :", train_mlp_y_data.shape)


# model.fit([train_mlp_x_data, train_cnn_data], train_mlp_y_data, epochs=20, verbose=2)
model.fit({"mlp_input": train_mlp_x_data, "cnn_input": train_cnn_data}, train_mlp_y_data, epochs=20, verbose=2)










