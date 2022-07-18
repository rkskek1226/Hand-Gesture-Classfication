import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, concatenate
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


seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)

Data_set = pd.read_csv("hand_gesture_data/hand_gesture_data2.csv")
x1 = Data_set.iloc[:, 20:45].astype(float)
y1 = Data_set.iloc[:, 45]

y1 = np_utils.to_categorical(y1)

x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.2, shuffle=True, stratify=y1, random_state=1)

mlp_input = Input(shape=(25, ))
mlp_hidden1 = Dense(64, activation="relu")(mlp_input)
mlp_hidden2 = Dense(64, activation="relu")(mlp_hidden1)
mlp_output = Dense(1, activation="softmax")(mlp_hidden2)
mlp_model = Model(inputs=mlp_input, outputs=mlp_output)

cnn_input = Input(shape=(640, 480, 3))
cnn_hidden1 = Conv2D(64, (3, 3), padding="same", activation="relu")(cnn_input)
cnn_hidden1 = MaxPooling2D((2, 2), strides=2)(cnn_hidden1)
cnn_hidden2 = Conv2D(128, (3, 3), padding="same", activation="relu")(cnn_hidden1)
cnn_hidden2 = MaxPooling2D((2, 2), strides=2)(cnn_hidden2)(cnn_hidden1)
cnn_hidden2 = Flatten()(cnn_hidden2)
cnn_hidden3 = Dense(128, activation="relu")(cnn_hidden2)
cnn_output = Dense(10, activation="softmax")(cnn_hidden3)
cnn_model = Model(input=cnn_input, outputs=cnn_output)

z = concatenate([mlp_input, cnn_output])
z = Dense(10, activation="relu")(z)   # 크게(보통 4배)      mlp에서도 4개로 늘리기
z_output = Dense(10, activation="softmax")(z)

model = Model(inputs=[mlp_input, cnn_input], outputs=z_output)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit()







