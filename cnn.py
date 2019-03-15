import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from tqdm import tqdm
from PIL import Image
import seaborn as sns
import keras
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.applications.vgg19 import VGG19
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
import cv2

IMG_SIZE = 90
TRAIN_FOLDER = "./train/"
TEST_FOLDER = "./test/"

sns.set(style="whitegrid")

def import_labels():
    labels = pd.read_csv("labels.csv")
    test_labels = (pd.read_csv("sample_submission.csv"))['id']
    return labels, test_labels

def plot_breed_distribution():

    df = (labels.groupby(['breed']).count().sort_values(by=['id'], ascending=False)).reset_index()
    f, ax = plt.subplots(figsize=(6, 25))
    tick_spacing = 2
    sns.barplot(x=df['id'], y=df['breed'], data=df,label="Total", color="b")
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.tick_params(axis='y', labelsize=8, rotation=20)
    f.suptitle('Distribution of Dog breeds', fontsize=20)
    plt.xlabel('Count', fontsize=18)
    plt.ylabel('Breed', fontsize=16)
    plt.subplots_adjust(left=0.3)
    plt.show()

def one_hot_encoding():

    original_labels = pd.Series(labels['breed'])
    one_hot_labels = np.asarray(pd.get_dummies(original_labels, sparse = True))
    return one_hot_labels


def prepare_image_data_and_labels():

    train_images = []
    train_labels = []
    test_images = []
    i = 0

    for f, breed in tqdm(labels.values):
        img = cv2.imread(TRAIN_FOLDER + '{}.jpg'.format(f))
        label = one_hot_labels[i]
        train_images.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))
        train_labels.append(label)
        i += 1

    for f in tqdm(test_labels.values):
        img = cv2.imread(TEST_FOLDER + '{}.jpg'.format(f))
        test_images.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))

    #Cv2 need ths output in the correct form uint8 for the one-hot labels
    train_labels = np.array(train_labels, np.uint8)
    train_images = np.array(train_images, np.float32) / 255.
    test_images  = np.array(test_images, np.float32) / 255.

    print(train_images.shape)
    print(train_labels.shape)
    print(test_images.shape)

    return train_labels, train_images, test_images

def test_train_split():
    # Uses sklearns library to split the data
    x_train, x_validation, y_train, y_validation = train_test_split(train_images, train_labels, test_size=0.2, stratify=np.array(train_labels))
    print (x_train.shape)
    print (x_validation.shape)
    return x_train, x_validation, y_train, y_validation

def setup_and_run_cnn():
    # The base model uses VGG19 - a pre-trained model included with keras! The weights
    # are included from 'imagenet'
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

    # Add a new top layer, softmax will return values as close to 1 and 0.
    x = base_model.output
    x = Flatten()(x)
    predictions = Dense(120, activation='softmax')(x)

    # This is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # First: train only the top layers (which were randomly initialized)
    for layer in base_model.layers:
        layer.trainable = False

    # ADAM is an upgraded optimizer method, - Improved from the stochastic Gradient Descent method
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_acc', patience=3, verbose=1)]
    model.summary()

    model.fit(x_train, y_train, epochs=20, validation_data=(x_validation, y_validation), verbose=1)

labels, test_labels = import_labels()
one_hot_labels = one_hot_encoding()
train_labels, train_images, test_images = prepare_image_data_and_labels()
x_train, x_validation, y_train, y_validation = test_train_split()
setup_and_run_cnn()
