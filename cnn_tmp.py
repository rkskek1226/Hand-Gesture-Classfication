import datetime
import time
import os
import pathlib
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Resizing, BatchNormalization, Activation, Dropout
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers.experimental.preprocessing import Resizing, Rescaling

# 핸드폰 이미지 크기 width = 1080, height = 1440
# 640, 480 -> 160, 120

batch_size = 32
img_height = 160
img_width = 120


train_data = tf.keras.utils.image_dataset_from_directory("hand_gesture_data/data5/", labels="inferred",
                                                         label_mode="int",
                                                         batch_size=batch_size, image_size=(img_height, img_width),
                                                         shuffle=True, seed=1, validation_split=0.2, subset="training")
test_data = tf.keras.utils.image_dataset_from_directory("hand_gesture_data/data5/", labels="inferred", label_mode="int",
                                                        batch_size=batch_size, image_size=(img_height, img_width),
                                                        shuffle=True, seed=1, validation_split=0.2, subset="validation")

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_data = train_data.cache().prefetch(buffer_size=AUTOTUNE)
test_data = test_data.cache().prefetch(buffer_size=AUTOTUNE)

model = Sequential()
model.add(Rescaling(1. / 255))
model.add(Conv2D(32, (27, 27), padding="same", input_shape=(img_height, img_width, 3)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.25))
model.add(MaxPooling2D((2, 2), strides=1))
model.add(Conv2D(128, (27, 27), padding="same"))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.25))
model.add(MaxPooling2D((2, 2), strides=1))
model.add(Conv2D(64, (27, 27), padding="same"))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.25))
model.add(MaxPooling2D((2, 2), strides=1))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(10, activation="softmax"))

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

start = time.time()

# history = model.fit(train_data, epochs=20, validation_data=test_data, verbose=2)
history = model.fit(train_data, epochs=20, validation_data=test_data, verbose=2)

sec = time.time() - start
time = str(datetime.timedelta(seconds=sec)).split(".")
times = time[0]
print("\ntime =", time)

model.save("qqqqqqqqqqqqqq.h5")

plt.subplot(211)
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend(["train", "validation"], loc="lower right")

plt.subplot(212)
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(["train", "validation"], loc="upper right")
plt.tight_layout()
plt.show()
