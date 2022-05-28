import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, GlobalMaxPool2D, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator









BATCH_SIZE = 20
image_width = 450
image_height = 960

train = ImageDataGenerator(rescale=1. / 255, rotation_range=10, width_shift_range=0.1,
                           height_shift_range=0.1, shear_range=0.1, zoom_range=0.2)
train_generator = train.flow_from_directory("hand_gesture_data/data3/train", target_size=(image_height, image_width),
                                            color_mode="rgb", batch_size=BATCH_SIZE, seed=1, shuffle=True,
                                            class_mode="categorical")

test = ImageDataGenerator(rescale=1.0 / 255.0)
test_generator = test.flow_from_directory("hand_gesture_data/data3/test", target_size=(image_height, image_width),
                                          color_mode="rgb", batch_size=BATCH_SIZE, seed=2, shuffle=True,
                                          class_mode="categorical")

history = model.









