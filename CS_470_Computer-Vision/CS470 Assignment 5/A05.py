import numpy as np
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten


def get_approach_names():
    approach_name = "kNN", "NN"
    return approach_name

def is_keras_model(approach_name):
    if(approach_name == "NN"):
        return True

    return False

def prepare_data(approach_name, data):
    match approach_name:
        case "kNN":
            data = data.astype("float32")
            data = np.reshape(data, (data.shape[0], -1))
            data /= 255
            data -= 0.5
            data *= 2.0
            return data
        case "NN":
            data = data.astype("float32")
            data /= 255
            data -= 0.5
            data *= 2.0
            return data

def train_classifier(approach_name, x_train, y_train, y_hot_train, class_cnt):
    match approach_name:
        case "kNN":
            classifier = KNeighborsClassifier(n_neighbors=4, weights="uniform")
            classifier.fit(x_train, y_train)
            return classifier
        case "NN":
            model = Sequential()
            model.add(Conv2D(filters=32, kernel_size=4, activation="relu", input_shape=x_train.shape[1:]))
            model.add(Conv2D(filters=32, kernel_size=4, activation="relu"))
            model.add(MaxPooling2D(pool_size=2))

            model.add(Conv2D(filters=64, kernel_size=4, activation='relu'))
            model.add(Conv2D(filters=64, kernel_size=4, activation='relu'))
            model.add(MaxPooling2D(pool_size=2))

            model.add(Flatten())
            model.add(Dense(10, activation="relu"))
            model.add(Dense(class_cnt, activation="softmax"))

            model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            model.summary()
            model.fit(x_train, y_hot_train, epochs=100, batch_size=32)
            return model