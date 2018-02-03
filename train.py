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

train_data_dir = 'data/train'
test_data_dir = 'data/test'
train_labels_path = "data/train.csv"
df = pd.read_csv(train_labels_path)

batch_size = 16
epochs = 100


def read_random_image(image_path, resize_size):
    random_sample = df.sample(n=1)
    img_name = random_sample['Image'].iloc[0]
    img = cv2.imread(
        image_path + "/" + img_name)
    img = cv2.resize(img, (resize_size, resize_size))
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_image = gray_image.astype(np.float32)
    return gray_image, img_name


def import_labels():
    """ This function import labels from csv / Create a list of unique
    labels and a dict with image_name and its label
    Output : dict('image_name': 'label')
    """

    dict_labels = df.set_index('Image').to_dict()['Id']
    unique_labels = sorted(list(set(dict_labels.values())))
    for index, label in dict_labels.items():
        dict_labels[index] = unique_labels.index(label)
    return dict_labels, unique_labels


def data_generator(batch_size, dict_labels, unique_labels, rezise_size):
    while True:
        x_train = []
        y_train = []
        for i in range(batch_size):
            img, img_name = read_random_image(train_data_dir, rezise_size)
            x_train.append(img)
            y_train.append(dict_labels[img_name])
        y_train = np_utils.to_categorical(y_train, len(unique_labels))
        x_train = np.array(x_train, dtype="float") / 255.0
        x_train = x_train.reshape(batch_size, rezise_size, rezise_size, 1)
        yield x_train, y_train


if __name__ == "__main__":
    dict_labels, unique_labels = import_labels()
    rezise_size = 228
    model = LeNet.build(width=rezise_size, height=rezise_size,
                        depth=1, nb_classes=len(unique_labels))
    model.compile(loss="categorical_crossentropy", optimizer="adam",
                  metrics=["accuracy"])

    model.fit_generator(data_generator(batch_size, dict_labels, unique_labels, rezise_size),
                        samples_per_epoch=1000, nb_epoch=20,
                        validation_data=data_generator(batch_size, dict_labels, unique_labels, rezise_size),
                        validation_steps=10)
    '''model = LeNet.build(width=x_train.shape[1], height=x_train.shape[2],
                        depth=x_train.shape[3], nb_classes=y_train.shape[1])
    model.compile(loss="categorical_crossentropy", optimizer="adam",
                  metrics=["accuracy"])'''
    '''image_dict = import_images(train_data_dir, 128)
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
    del model'''
