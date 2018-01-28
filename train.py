import pandas as pd
import numpy as np
import os
import cv2
from lenet import LeNet
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import keras.backend as K
from keras.optimizers import Adam

# https://www.pyimagesearch.com/2017/12/11/image-classification-with-keras-and-deep-learning/#

config = K.tf.ConfigProto()
config.gpu_options.allow_growth = True
session = K.tf.Session(config=config)

train_data_dir = 'data/train'
test_data_dir = 'data/test'
train_labels_path = "data/train.csv"

batch_size = 64
epochs = 100


def import_images(image_folder, resize_size):
    """ Import images from a folder
    Output : dict('image_name': IMAGE (List of pixels))
    """
    image_dict = {}
    i = 0
    for element in os.listdir(image_folder):
        img = cv2.imread(
            image_folder + "/" + element)
        img = cv2.resize(img, (resize_size, resize_size))
        img = img.astype(np.float32)
        image_dict[element] = img
    return image_dict


def import_labels(label_path):
    """ This function import labels from csv / Create a list of unique
    labels and a dict with image_name and its label
    Output : dict('image_name': 'label')
    """
    labels_df = pd.read_csv(label_path)
    print(labels_df.head())
    dict_labels = labels_df.set_index('Image').to_dict()['Id']
    unique_labels = sorted(list(set(dict_labels.values())))
    for index, label in dict_labels.items():
        dict_labels[index] = unique_labels.index(label)
    return dict_labels, unique_labels


if __name__ == "__main__":
    dict_labels, unique_labels = import_labels(train_labels_path)
    image_dict = import_images(train_data_dir, 128)
    # print(image_dict["0042ea34.jpg"].shape)
    x_train = []
    y_train = []
    for image_name, image in image_dict.items():
        x_train.append(image)
        y_train.append(dict_labels[image_name])
    y_train = np_utils.to_categorical(y_train, len(unique_labels))
    x_train = np.array(x_train, dtype="float") / 255.0
    model = LeNet.build(width=x_train.shape[1], height=x_train.shape[2],
                        depth=x_train.shape[3], nb_classes=y_train.shape[1])
    model.compile(loss="categorical_crossentropy", optimizer="adam",
                  metrics=["accuracy"])
    model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs,
              verbose=2, validation_split=0.2)
    # Save model
    model.save('basic_model.h5')
    del model
