import pandas as pd
import numpy as np
import os
import cv2

train_data_dir = 'data/train'
test_data_dir = 'data/test'
train_labels_path = "data/train.csv"


def import_images(image_folder, resize_size):
    """ Import images from a folder
    Output : dict('image_name': IMAGE (List of pixels))
    """
    image_dict = {}
    i = 0
    for element in os.listdir(image_folder):
        if i > 100:
            break
        else:
            img = cv2.imread(
                image_folder + "/" + element)
            img = cv2.resize(img, (resize_size, resize_size))
            img = img.astype(np.float32)
            image_dict[element] = img
            i += 1
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
    return dict_labels, unique_labels


if __name__ == "__main__":
    dict_labels, unique_labels = import_labels(train_labels_path)
    image_dict = import_images(train_data_dir, 227)
    print(image_dict["0042ea34.jpg"].shape)
    x_train = []
    y_train = []
    for image_name, image in image_dict.items():
        x_train.append(image)
        y_train.append(dict_labels[image_name])
    print(x_train, y_train)
