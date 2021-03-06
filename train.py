import pandas as pd
import numpy as np
import os
import cv2
from lenet import LeNet
import random
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16
from sklearn.model_selection import train_test_split
from keras.layers import Input
from keras.utils import plot_model
from keras.layers.core import Flatten
from image_processing import random_crop
from image_processing import blur
from image_processing import random_rotate_zoom
from image_processing import vertical_flip

# https://www.pyimagesearch.com/2017/12/11/image-classification-with-keras-and-deep-learning/#

train_data_dir = 'data/train'
custom_train_data_dir = 'data/custom_train'
test_data_dir = 'data/test'
train_labels_path = "data/train.csv"

# Remove image not in custom_train
custom_df = pd.read_csv(train_labels_path)
custom_train_images_set = set()
for element in os.listdir(custom_train_data_dir):
    custom_train_images_set.add(element)

custom_df = custom_df[custom_df['Image'].isin(custom_train_images_set)]
custom_df_train, custom_df_test = train_test_split(custom_df, test_size=0.2)

batch_size = 16
epochs = 50


def read_random_image(image_path, resize_size, train):
    if train:
        random_sample = custom_df_train.sample(n=1)
        img_name = random_sample['Image'].iloc[0]
        gray_img = cv2.imread(
            image_path + "/" + img_name, cv2.IMREAD_GRAYSCALE)
        crop_chance = random.random()
        blur_chance = random.random()
        rotate_zoom_chance = random.random()
        flip_chance = random.random()
        if crop_chance <= 1:
            gray_img = random_crop(gray_img)
        if blur_chance <= 1:
            gray_img = blur(gray_img, random.choice([3, 5]))
        if rotate_zoom_chance <= 1:
            gray_img = random_rotate_zoom(gray_img)
        if flip_chance <= 1:
            gray_img = vertical_flip(gray_img)
    else:
        random_sample = custom_df_test.sample(n=1)
        img_name = random_sample['Image'].iloc[0]
        gray_img = cv2.imread(
            image_path + "/" + img_name, cv2.IMREAD_GRAYSCALE)
    gray_img = cv2.resize(gray_img, (resize_size, resize_size))
    back_to_rgb_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
    return back_to_rgb_img, img_name


def import_labels():
    """ This function import labels from csv / Create a list of unique
    labels and a dict with image_name and its label
    Output : dict('image_name': 'label')
    """

    dict_labels = custom_df.set_index('Image').to_dict()['Id']
    unique_labels = sorted(list(set(dict_labels.values())))
    for index, label in dict_labels.items():
        dict_labels[index] = unique_labels.index(label)
    return dict_labels, unique_labels


def data_generator(batch_size, dict_labels, unique_labels, rezise_size, train):
    while True:
        x_train = []
        y_train = []
        for i in range(batch_size):
            img, img_name = read_random_image(
                custom_train_data_dir, rezise_size, train)
            x_train.append(img)
            y_train.append(dict_labels[img_name])
        y_train = np_utils.to_categorical(y_train, len(unique_labels))
        x_train = np.array(x_train, dtype="float") / 255.0
        x_train = x_train.reshape(batch_size, rezise_size, rezise_size, 3)
        yield x_train, y_train


if __name__ == "__main__":
    dict_labels, unique_labels = import_labels()
    resize_size = 224
    input_tensor = Input(shape=(resize_size, resize_size, 3))
    vgg_model = VGG16(input_tensor=input_tensor, weights="imagenet", include_top=False)
    # base_model.layers.pop()

    # add a global spatial average pooling layer
    x = vgg_model.output
    x = Flatten()(x)
    # let's add a fully-connected layer
    x = Dense(512, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(len(unique_labels), activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=vgg_model.input, outputs=predictions)
    for layer in model.layers[:19]:
        layer.trainable = False
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit_generator(data_generator(batch_size, dict_labels, unique_labels, resize_size, True),
                        samples_per_epoch=250, nb_epoch=epochs,
                        validation_data=data_generator(
                            batch_size, dict_labels, unique_labels, resize_size, False),
                        validation_steps=60)
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
