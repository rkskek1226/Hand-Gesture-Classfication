import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
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
x1 = Data_set.iloc[:, :45].astype(float)
y1 = Data_set.iloc[:, 45]

y1 = np_utils.to_categorical(y1)

x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.2, shuffle=True, stratify=y, random_state=1)








