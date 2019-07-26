import keras
import os
import numpy as np
from all_config import *
from image_utils.image_utils import *
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras import regularizers


# input image dimensions
img_rows, img_cols = IMG_ROW, IMG_COL
trained_model = model_name


class Image(object):
    """
    class for generate train image from user image in image folder
    """
    def __init__(self):
        self.image_path = IMAGE_PATH
        self.train_path = TRAIN_PATH

    def get_augmentation(self):
        """
        generate train image from image folder
        :return:
        """
        cls_list = ["0", "1"]
        self._clear_train_path()
        for cls_prefix in cls_list:
            x = []
            i = 0
            for name in os.listdir(self.image_path):
                cls_num = name.split("_")[0]
                if cls_num == cls_prefix:
                    img = cv2.imread(self.image_path + name)
                    _, img = get_binary_image(img)
                    img = cv2.resize(img, (img_rows, img_cols))
                    x.append(img)
            x = np.asarray(x, np.float32)
            data_gen = ImageDataGenerator(rotation_range=20, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True,
                                          vertical_flip=True, fill_mode='nearest', data_format='channels_last')
            for _ in data_gen.flow(x, batch_size=1, save_to_dir=self.train_path, save_prefix=cls_prefix, save_format="png"):
                i = i+1
                if i >= augmentation_size:
                    print("class_{0} augmentation for {1} samples".format(cls_prefix, i))
                    break

    def _clear_train_path(self):
        """
        clear train path
        :return:
        """
        if os.path.exists(self.train_path):
            for file in os.listdir(self.train_path):
                os.remove(self.train_path+file)
        else:
            os.mkdir(self.train_path)


def get_data():
    """
    get train data from train folder and transfer in numpy type
    :return: train and test in numpy type
    """
    x = []
    y = []
    for name in os.listdir(TRAIN_PATH):
        cls_num = name.split("_")[0]
        img = cv2.imread(TRAIN_PATH + name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (img_rows, img_cols))
        x.append(img)
        y.append(cls_num)
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    return train_test_split(x, y, random_state=30, test_size=.28)


def train_model():
    """
    build model and compile with config params
    :return:
    """
    # the data, split between train and test sets
    x_train, x_test, y_train, y_test = get_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(0.01)))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    model.save(trained_model)
    print(model.summary())
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    # history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


"""
if you want update train image, run Image().get_augmentation,
you will get a trained model by train_model()
"""
if __name__ == "__main__":
    Image().get_augmentation()
    train_model()
