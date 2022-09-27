import tensorflow as tf
from keras.layers import Conv2D, BatchNormalization, Activation, Add, MaxPooling2D, GlobalAveragePooling2D, Dense, Flatten

#
# class IdentityBlock(tf.keras.Model):
#     def __init__(self, filters, filter_size):
#         super(IdentityBlock, self).__init__()
#         self.conv1 = Conv2D(filters, (filter_size, filter_size), padding="same")
#         self.bn1 = BatchNormalization()
#
#         self.conv2 = Conv2D(filters, (filter_size, filter_size), padding="same")
#         self.bn2 = BatchNormalization()
#
#         self.act = Activation("relu")
#         self.add = Add()
#
#     def call(self, inputs):
#         x = self.conv1(inputs)
#         x = self.bn1(x)
#         x = self.act(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         # ??
#
#         x = self.add([x, inputs])
#         x = self.act(x)
#         return x
#
#
# class Resnet18(tf.keras.Model):
#     def __init__(self, num_classes):
#         super(Resnet18, self).__init__()
#         self.conv = Conv2D(64, (7, 7), padding="same")
#         self.bn = BatchNormalization()
#         self.act = Activation("relu")
#         self.max_pool = MaxPooling2D((3, 3))
#         self.ib1a = IdentityBlock(64, 3)
#         self.ib1b = IdentityBlock(64, 3)
#
#         self.gap = GlobalAveragePooling2D()
#         self.classifier = Dense(num_classes, activation="softmax")
#
#     def call(self, inputs):
#         x = self.conv(inputs)
#         x = self.bn(x)
#         x = self.act(x)
#         x = self.max_pool(x)
#
#         x = self.ib1a(x)
#         x = self.ib1b(x)
#
#         x = self.gap(x)
#         return self.classifier(x)



# Input tensor
x = tf.keras.Input((224, 224, 3))
# conv_1
conv1 = Conv2D(64, (7, 7), padding="same", strides=2)(x)
conv1 = BatchNormalization()(conv1)
conv1 = Activation("relu")(conv1)
conv1 = MaxPooling2D((3, 3), padding="SAME", strides=2)(conv1)

# conv_2_x
conv2_1 = Conv2D(64, (3, 3), padding="same", strides=1)(conv1)
conv2_1 = BatchNormalization()(conv2_1)
conv2_1 = Activation("relu")(conv2_1)
conv2_1 = Conv2D(64, (3, 3), padding="same", strides=1)(conv2_1)
conv2_1 = BatchNormalization()(conv2_1)
short_cut = Conv2D(64, (1, 1), padding="same", strides=1)(conv1)
conv2_1 = tf.keras.layers.add([conv2_1, short_cut])
conv2_1 = Activation("relu")(conv2_1)

conv2_2 = Conv2D(64, (3, 3), padding="same", strides=1)(conv2_1)
conv2_2 = BatchNormalization()(conv2_2)
conv2_2 = Activation("relu")(conv2_2)
conv2_2 = Conv2D(64, (3, 3), padding="same", strides=1)(conv2_2)
conv2_2 = BatchNormalization()(conv2_2)
conv2_2 = tf.keras.layers.Add()([conv2_2, conv2_1])
conv2_2 = Activation("relu")(conv2_2)

# conv_3_x
conv3_1 = Conv2D(128, (3, 3), padding="same", strides=2)(conv2_2)
conv3_1 = BatchNormalization()(conv3_1)
conv3_1 = Activation("relu")(conv3_1)
conv3_1 = Conv2D(128, (3, 3), padding="same", strides=1)(conv3_1)
conv3_1 = BatchNormalization()(conv3_1)
short_cut = Conv2D(128, (1, 1), padding="same", strides=2)(conv2_2)
conv3_1 = tf.keras.layers.Add()([conv3_1, short_cut])
conv3_1 = Activation("relu")(conv3_1)

conv3_2 = Conv2D(128, (3, 3), padding="same", strides=1)(conv3_1)
conv3_2 = BatchNormalization()(conv3_2)
conv3_2 = Activation("relu")(conv3_2)
conv3_2 = Conv2D(128, (3, 3), padding="same", strides=1)(conv3_2)
conv3_2 = BatchNormalization()(conv3_2)
conv3_2 = tf.keras.layers.Add()([conv3_2, conv3_1])
conv3_2 = Activation("relu")(conv3_2)

# conv_4_x
conv4_1 = Conv2D(256, (3, 3), padding="same", strides=2)(conv3_2)
conv4_1 = BatchNormalization()(conv4_1)
conv4_1 = Activation("relu")(conv4_1)
conv4_1 = Conv2D(256, (3, 3), padding="same", strides=1)(conv4_1)
conv4_1 = BatchNormalization()(conv4_1)
short_cut = Conv2D(256, (1, 1), padding="same", strides=2)(conv3_2)
conv4_1 = tf.keras.layers.Add()([conv4_1, short_cut])
conv4_1 = Activation("relu")(conv4_1)

conv4_2 = Conv2D(256, (3, 3), padding="same", strides=1)(conv4_1)
conv4_2 = BatchNormalization()(conv4_2)
conv4_2 = Activation("relu")(conv4_2)
conv4_2 = Conv2D(256, (3, 3), padding="same", strides=1)(conv4_2)
conv4_2 = BatchNormalization()(conv4_2)
conv4_2 = tf.keras.layers.Add()([conv4_2, conv4_1])
conv4_2 = Activation("relu")(conv4_2)

# conv_5_x
conv5_1 = Conv2D(512, (3, 3), padding="same", strides=2)(conv4_2)
conv5_1 = BatchNormalization()(conv5_1)
conv5_1 = Activation("relu")(conv5_1)
conv5_1 = Conv2D(512, (3, 3), padding="same", strides=1)(conv5_1)
conv5_1 = BatchNormalization()(conv5_1)
short_cut = Conv2D(512, (1, 1), padding="same", strides=2)(conv4_2)
conv5_1 = tf.keras.layers.Add()([conv5_1, short_cut])
conv5_1 = Activation("relu")(conv5_1)

conv5_2 = Conv2D(512, (3, 3), padding="same", strides=1)(conv5_1)
conv5_2 = BatchNormalization()(conv5_2)
conv5_2 = Activation("relu")(conv5_2)
conv5_2 = Conv2D(512, (3, 3), padding="same", strides=1)(conv5_2)
conv5_2 = BatchNormalization()(conv5_2)
conv5_2 = tf.keras.layers.Add()([conv5_2, conv5_1])
conv5_2 = Activation("relu")(conv5_2)

avg_pool = GlobalAveragePooling2D()(conv5_2)
flat = Flatten()(avg_pool)
dense10 = Dense(10, activation="softmax")(flat)

model = tf.keras.Model(inputs=x, outputs=dense10)

print(model.summary())




